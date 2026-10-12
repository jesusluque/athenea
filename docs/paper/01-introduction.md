<!-- Copyright (c) 2026 jesus luque. -->

# 1. Introduction

## 1.1 The problem

Production 3D assets are authored as meshes: polygons, layered physically based materials, textures, lights and,
for characters, skeletons, all composed in USD. They are rendered faithfully by a path tracer, which is too slow for a
phone, and interactively by a rasteriser, which needs the asset's full geometry and texture set resident and still
approximates its light transport. 3D Gaussian splatting (3DGS) offers a third route: a cloud of anisotropic Gaussians,
sorted and blended in screen space, draws photographic detail at interactive rates on modest hardware. A Gaussian
cloud, however, is normally *trained* from photographs of a real scene. It holds the light of that capture, it cannot
be relit, and it has no material, no skeleton and no relation to an authored asset.

This paper addresses the converse problem. Given a photoreal USD asset (the case followed throughout is a Chevrolet
Corvette C7 of 192 meshes with OpenPBR car paint, chrome, glass, rubber and lamps), produce a Gaussian representation
that

1. is **faithful to a path trace** of the source mesh, judged image against image and region by region;
2. **relights**: it answers any sky and any light, not only the illumination it was made under;
3. keeps a rig where the asset has one, so that an animated asset still moves;
4. **plays back on a phone** inside a web page, within a fixed memory budget, close up as well as far away.

Neither of the obvious routes meets all four. Training a cloud from path-traced views of the asset gives a fixed
illumination and spends hours of optimisation per asset. Rasterising the mesh on the phone is limited by memory and
by the shading a mobile GPU can afford per pixel. The route taken here converts the mesh directly into Gaussians,
bakes the asset's light transport into each Gaussian with the renderer's own path tracer, and then reduces, compresses
and streams the result.

## 1.2 Goals and constraints

The work was carried out under a small set of constraints, set by the user and held throughout. They shape almost
every design decision in Sections 2–7.

| constraint | statement | where it acts |
|---|---|---|
| GPU-only arithmetic | No arithmetic on scene data runs on the CPU: no CPU renderer, fallback or test oracle. The host reads files and keeps the books (counts, slots, queues). | Every section; §2.1 states the rule |
| Pure 3DGS at playback | What the player draws is Gaussians only. The ground is not a mesh: its interaction with the object is converted into Gaussians (a shadow catcher). A mesh is only an input to the bake. | §3.9, §7.9 |
| Fidelity first | The reference is the mesh path trace. Where quality and frame rate conflict, quality wins ("aunque empeore el rendimiento quiero calidad", 12 October 2026). | §4.10, §7.10 |
| Memory budget | 50–60 MB of resident car data while the camera moves (about 55 MB in the current player), about 100 MB at rest; render state is counted apart. | §6.8, §7.10, §7.11 |
| Automatic, not manual | Per-asset choices are detected from the data: the cell size from a camera, solid versus thin glass, closed shells, the ground for the shadow catcher. Options only override. | §3.3, §3.5, §3.8, §3.9 |
| Small pieces must look right | Badges, grilles, trims and lamps are judged as carefully as panels, close up as well as far. | §3.11, §6.9, §7.12 |

## 1.3 Contributions

The contributions described in this paper are the following.

1. **A GPU-only USD renderer** (athenea, Section 2) that draws meshes, points and Gaussian splats through Hydra, with
   a tile rasteriser and a Gaussian ray tracer that share one Gaussian layout, a MaterialX-to-Slang material compiler
   checked against MaterialX itself, a path tracer with multiple importance sampling used as ground truth, and a
   verification discipline in which every image comparison is itself a GPU kernel.
2. **A mesh-to-Gaussian conversion** (mesh2splat, Section 3), run as an AOFX effect on the device, with cells derived
   from a camera's pixel footprint, opacity that reproduces a surface's coverage, hard edges and closed shells found on
   the device, Toksvig widening of roughness by the normals' spread, layered OpenPBR capture, and automatic glass
   classification.
3. **A relightable transfer baked into each Gaussian** (TX, Section 4): degree-3 spherical harmonics of direct and
   indirect transfer, a reflected field normalised over closed directions and coupled to the sky at draw time, and
   256 bits of open directions read as cones by each reflection lobe; per-pixel sharp lobes driven by a stored shape
   operator, with the bake's widening undone and a tube distinguished from a fold.
4. **Skinned Gaussian clouds** (Section 5) that carry the rig rather than the frames: a skinning kernel using the full
   Jacobian including the weights' gradients, zonal transfer lobes that turn with the Gaussian, per-part visibility
   fields read as a product of table lookups, and decimation that keeps the rig.
5. **Per-part merge trees with a bounded error cut** (Section 6): pair merging under guards that respect parts and
   materials, a least-squares coverage opacity for merges, an error made monotone with covering spheres so that one
   parallel test selects the cut, back-face culling of closed shells, a conservative shared occlusion codebook, and
   pages ordered by the parent's error.
6. **A compression and streaming path to a phone** (Section 7): a `.athc` v3 container with byte-plane encodings,
   clustered PCA of the transfer kept compressed on the GPU, vector-quantised open-direction bits, a 40-byte compact
   record, quantised traversal records of 5–11 bytes, and a frame-time and still-frame policy that keeps the car's
   data within its budget.
7. **A measured account** (Section 8) of what each step buys against the path trace, on one production asset, from the
   bake to a real iPhone.

## 1.4 Organisation

Section 2 describes the engine and renderer. Section 3 describes the conversion from USD meshes to Gaussians, and
Section 4 the light transport baked into them and how they are relit. Section 5 covers animated assets: skinning, the
sparrow and the pawn. Section 6 covers levels of detail, and Section 7 compression and the WebGPU player on the
iPhone. Section 8 consolidates the evaluation from Sections 2–7, Section 9 gathers the limitations and the work left,
and Section 10 lists the references, followed by an appendix on the repositories, commands and figures.

## 1.5 Related work

**Gaussian splatting.** Kerbl et al. (2023) introduced 3D Gaussian splatting: a scene as anisotropic Gaussians with
spherical-harmonic colour, optimised from photographs and drawn by a tile-based, depth-sorted rasteriser. Its
projection is the EWA splatting of Zwicker et al. (2001, 2002), a local affine approximation of the perspective map.
Mip-Splatting (Yu et al., 2024) adds a 3D smoothing filter and a 2D screen filter against aliasing; athenea's 0.3 px²
dilation with its opacity compensation is of that family (§2.5.1). 2D Gaussian splatting (Huang et al., 2024) flattens
each Gaussian to an oriented disc evaluated by ray–plane intersection, in the line of surfels (Pfister et al., 2000);
athenea's surfels follow it (§2.5.2). 3D Gaussian ray tracing (Moënne-Loccoz et al., 2024) traces particles through
bounding proxies in a BVH, which athenea's hardware route follows (§2.6).

**From meshes to Gaussians.** Electronic Arts' mesh2splat converts a mesh into Gaussians by rasterising each triangle
in its dominant axis plane, one Gaussian per covered texel; athenea's conversion starts from it (§3.1). Mesh2GS
(arXiv 2606.21898) derives a Gaussian's size from a camera's pixel footprint, which athenea's `--cell-from-camera`
follows (§3.3).

**Precomputed light transport.** Precomputed radiance transfer (Sloan, Kautz and Snyder, 2002) stores, per point, the
projection of visibility times the cosine onto spherical harmonics, so that relighting is a dot product with the
environment's coefficients; Ramamoorthi and Hanrahan (2001) showed that nine coefficients represent irradiance well.
Clustered PCA (Sloan et al., 2003) compresses such transfers by clustering and a per-cluster principal basis, which
athenea uses both for the file and on the phone's GPU (§7.2). Zonal harmonics, which rotate by rotating an axis, are
fitted after Sloan (2008) (§4.4.5, §5.7). Filtered importance sampling (Křivánek and Colbert, 2008) prefilters the
environment (§4.3), and Lagarde and de Rousiers (2014) give the specular occlusion fit that the first transfer used
(§4.6.1).

**Specular antialiasing.** Toksvig (2005) widens a microfacet distribution by the variance of the normals it
averages; athenea applies the same idea at conversion, to the spread of normals under one Gaussian (§3.7, §4.9).

**Levels of detail for Gaussians.** Octree-GS (Ren et al., 2024) organises anchors in an octree selected by view
distance. Hierarchical 3D Gaussians (Kerbl et al., 2024) build a merge hierarchy from Gaussian moments and cut it by
screen size; athenea's original level of detail follows its moments (§6.1). NanoGS (Xiong et al., 2026) merges
mutual nearest neighbours pairwise, which athenea's per-part trees run on the device (§6.2). The cost uses Runnalls'
(2007) bound for Gaussian mixture reduction, and the error-band pages follow Nanite's (Karis et al., 2021) use of a
monotone error for a parallel cut (§6.3, §6.8).

**Engines and formats.** The engine is built on OpenUSD and Hydra, MaterialX and OpenPBR materials, Slang shaders
compiled through slang-rhi to Metal, Vulkan and CUDA, Intel Open Image Denoise and OpenColorIO. The player is a fork
of Spark with a WebGPU backend.

## 1.6 Conventions

Unless stated otherwise, measurements were taken on an Apple M5 Pro under Metal. Images are compared on the GPU
(§2.15). **PSNR** is in dB on display-encoded images; **relMSE** is relative mean squared error on linear values.
**The path trace** (PT) of the Corvette is the mesh path-traced from `/World/Camera` at 1920 × 1080 under the
`autoshop_01_4k` sky. **Car crop** is PSNR on the 900 × 420 crop of the car, **frame** on the whole frame; a **class
ratio** is a region's mean over the path trace's. The validate-corvette harness of the player measures other regions
(car, door, sill, bumper, hood), which are not the car crop (§7.12.3). Test identifiers (T15–T75) are those of the
pieces work's log, and commit hashes are those of the repository named (Appendix A). The words *Gaussian* and *splat*
are used interchangeably; a *leaf* is a baked Gaussian, a *merge* a node of a merge tree, and a *cut* the set of nodes
drawn for one view. *Transfer* is the baked light transport (§4), *TX* its current form. Figures named in the
sections as file paths are listed in Appendix A.3.
