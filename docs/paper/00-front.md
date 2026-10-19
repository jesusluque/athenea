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

::: {.nota}
**Notes on the title and the abstract.** The notes that follow comment, in the order in which they appear, on the terms of the title and the abstract. They are placed here, after the `#front` block closes, so as not to alter the layout of the title page or the table of contents.

**Baker, bake.** In production, to *bake* is to precompute an expensive quantity —lighting, occlusion, normals, displacement— and store it in a medium that is cheap to read at draw time: textures (*lightmaps*, *AO maps*), vertices or, here, the Gaussians themselves. A *baker* is the program that does that precomputation. The key idea is to move the cost from draw time (every frame, in real time) to asset preparation time (once, without hurry), in exchange for memory and a loss of generality: what has been precomputed no longer changes if the scene changes. athenea's twist is that it does not bake the *light* (the final radiance) but the *transport* of light, which is independent of the sky; that is why the result can be relit.

**Relight.** To light an object again under an illumination other than the one it had when it was captured or precomputed. Mathematically, if outgoing radiance is linear in incident radiance $L_{\text{in}}$, there is a linear operator $\mathcal{T}$ (the transport) such that $L_{\text{out}} = \mathcal{T}\,L_{\text{in}}$. Storing an approximation of $\mathcal{T}$ per point allows it to be applied to any new $L_{\text{in}}$; storing only $L_{\text{out}}$ (as a 3DGS trained from photographs does) freezes a single illumination.

**Production USD.** OpenUSD (Universal Scene Description, Pixar, [43] in §10) is the scene format and composition system used by studios: a *stage* is composed of *layers* through *references*, *payloads*, *variants* and *overrides*; geometry is `UsdGeomMesh` *prims*, materials are `UsdShade` *networks* (here MaterialX/OpenPBR), lights are `UsdLux`, skeletons `UsdSkel`. "Production" means that the asset has not been prepared for the baker: it has hundreds of meshes, layered materials, UDIM textures, instances and transform hierarchies just as they come out of a DCC.

*References:* [43] in §10; specification at openusd.org ("Introduction to USD", "Glossary"); code: `modules/usd/src/MeshStage.cpp` (reading the stage for conversion), `modules/usd/src/Export.cpp` (writing the master).
:::

::: {.nota}
**3D Gaussian splatting (3DGS).** A representation of a scene as a set of anisotropic 3D Gaussians that are projected to the screen and blended in depth order (Kerbl et al. 2023, [1]). Each Gaussian $i$ is an unnormalised density function
$$G_i(\mathbf{x}) = \exp\!\left(-\tfrac12 (\mathbf{x}-\boldsymbol{\mu}_i)^{\mathsf T}\,\Sigma_i^{-1}\,(\mathbf{x}-\boldsymbol{\mu}_i)\right),$$
with centre $\boldsymbol{\mu}_i$, covariance $\Sigma_i$, an opacity $o_i\in[0,1]$ and a colour, which in 3DGS is a function of the view direction expressed in spherical harmonics.

**Anisotropic, covariance.** "Anisotropic" means that the Gaussian is not a sphere: its level ellipsoids have three axes of different lengths. The covariance is parameterised so that it is always symmetric positive definite: $\Sigma = R\,S\,S^{\mathsf T}R^{\mathsf T}$, with $R$ a rotation (stored as a quaternion) and $S=\mathrm{diag}(s_x,s_y,s_z)$ the scales (standard deviations along the principal axes). If one scale is almost zero, the Gaussian degenerates into a disc: this is what athenea calls a *surfel* (§2.5.2), and it is the natural shape of a Gaussian that represents a patch of a mesh's surface.

**Splat, splatting.** *Splatting* is the rendering technique of "splashing": instead of casting rays, each primitive is projected to the screen and deposits its *footprint* on the pixels it covers. A *splat* is that footprint and, by extension, the primitive. In EWA projection (Zwicker et al. [3]) the 3D Gaussian is transformed into a 2D Gaussian on the screen with covariance
$$\Sigma' = J\,W\,\Sigma\,W^{\mathsf T}J^{\mathsf T},$$
where $W$ is the linear part of the world-to-camera transform and $J$ the Jacobian of the perspective projection evaluated at the centre (the local affine approximation). At each pixel $\mathbf{p}$ the Gaussian contributes $\alpha_i = o_i \exp(-\tfrac12 \mathbf{d}^{\mathsf T}\Sigma'^{-1}\mathbf{d})$, $\mathbf{d}=\mathbf{p}-\boldsymbol{\mu}'_i$, and the colour is composited front to back:
$$C = \sum_i c_i\,\alpha_i \prod_{j<i}(1-\alpha_j).$$

**Why 3DGS is attractive here.** The representation is explicit (there is no neural network to evaluate), tile rasterisation is highly parallel and of predictable cost, and a single Gaussian can represent sub-pixel detail, partial transparency and soft edges. The price: depth order is per Gaussian (not per pixel), which produces *popping* when two Gaussians swap order, and the memory per primitive is large (tens to hundreds of bytes with SH).

*References:* [1], [3] in §10; code: `shaders/athenea/common/packing.slang` (`encodeQuaternion`, `axesOfQuaternion`, `isSurfel`), `shaders/athenea/splat/splat_project.slang` (EWA projection, `splatProject`), `shaders/athenea/splat/splat_blend.slang` (`blendPixel`, tile compositing), `modules/scene/include/athenea/scene/GpuClouds.h` (`GpuSplats`, the shared cloud layout); `docs/decisions.md`, section "Ray tracing Gaussians".
:::

::: {.nota}
**A path trace as the reference.** *Path tracing* (Kajiya, "The Rendering Equation", SIGGRAPH 1986) solves the rendering equation
$$L_o(\mathbf{x},\omega_o) = L_e(\mathbf{x},\omega_o) + \int_{\Omega} f_r(\mathbf{x},\omega_i,\omega_o)\,L_i(\mathbf{x},\omega_i)\,(\mathbf{n}\cdot\omega_i)\,d\omega_i$$
by Monte Carlo: random paths are traced from the camera, bouncing off surfaces according to their BSDF $f_r$, and averaged. With $N$ paths the estimator is unbiased and its standard error decreases as $1/\sqrt{N}$. It is the reference because, with enough samples, it converges to the physical solution of the scene model (geometry, materials, lights) without shading approximations. In the paper, "faithful to the path trace" means that the image of the relit Gaussians is compared pixel by pixel with the image the same engine produces by tracing the original mesh under the same lighting.

**The source mesh.** The comparison is with the mesh, not with a cloud: that way the whole error introduced by the pipeline (conversion + bake + relight + LOD) is measured, not only that of one step.

*References:* [21] in §10 (MIS); Kajiya (1986); Pharr, Jakob and Humphreys, *Physically Based Rendering*, 4th ed. (2023), chs. 2 and 13; code: `modules/technique/src/PathTracer.cpp` (`PathTracer::trace`, `PathTracer::bake`); `docs/decisions.md`, section "Complete USD: the path tracer (M6)".
:::

::: {.nota}
**Conversion on the device.** *Device* is the GPU APIs' term (Metal `MTLDevice`, Vulkan `VkDevice`, CUDA context) for the GPU and its memory, as opposed to the *host* (the CPU). "On the device" means that the data are uploaded once and all computation is done with compute kernels on GPU buffers.

**One Gaussian per cell of a camera-derived lattice.** Each triangle is traversed over a regular grid (*lattice*) in the plane of two of its axes, and each covered *cell* emits one Gaussian. The cell size is derived from the camera that will look at it: a part's cell is what one pixel covers where the part is closest to the camera,
$$\text{cell} = z\cdot\frac{\text{aperture}/\text{pixels}}{\text{focal}},$$
with $z$ the distance from the camera to the part's box. This is Mesh2GS's rule [13]: one Gaussian per projected pixel gives every part the "right" density.

**Coverage-correct opacity.** If the surface covers a fraction $F$ (1 for an opaque surface, less with an opacity map or on glass), the opacity $\alpha$ of each Gaussian cannot simply be $F$: neighbouring Gaussians overlap, and the accumulated transmittance is $T=\prod(1-\alpha g)$. Taking $-\ln T \approx \sum \alpha g + \tfrac12\alpha^2 g^2$ over a lattice, with $\sum g = 2\pi\sigma^2$ and $\sum g^2 = \pi\sigma^2$ per cell (in cell units), the equation to solve is
$$\tfrac{\pi\sigma^2}{2}\,\alpha^2 + 2\pi\sigma^2\,\alpha = -\ln(1-F),$$
whose positive root is used, plus $1/255$ for the Gaussian's tail that is not drawn.

**Edges held.** A Gaussian is one cell wide; where a surface stops (an open border) or bends (a crease or *fold*, an edge with a large dihedral angle) a wide Gaussian sticks out. The conversion detects those edges on the device and reduces the Gaussian's variance in the direction perpendicular to the line, down to its distance from it.

*References:* [12], [13] in §10; code: `shaders/athenea/usd/mesh2splat_cells.slang` (the cell from a camera), `plugins/mesh2splat/mesh2splat.slang` (`m2sCoverageAlpha`, `m2sHoldEdges`, `projectionAxes`), `shaders/athenea/usd/mesh2splat_edges.slang`; `docs/decisions.md`, sections "A converted surface covers what its opacity says" and "mesh2splat: a mesh becomes a cloud, through the plugin interface".
:::

::: {.nota}
**Layered OpenPBR.** OpenPBR Surface ([30]) is the Academy Software Foundation's standard "uber-shader" material model, successor to Autodesk Standard Surface and Adobe Standard Material. It describes a surface as a stack of physical layers: *fuzz* (sheen), *coat* (a dielectric varnish with its own IOR and roughness), *specular* over a *base* that may be diffuse dielectric, metallic, transmissive or subsurface, with *thin-film*. Each layer attenuates what it lets through to the one below according to its Fresnel. A car paint is typically a metallic/flake base + coat; athenea captures per Gaussian the weight and parameters of each lobe (specular, coat, sheen) to evaluate them at draw time.

**Solid glass versus thin wall.** A *solid* dielectric (a ball, a solid lamp lens) refracts: the ray enters, bends by Snell's law, crosses the volume and leaves through another face. A *thin wall* (*thin-walled*: a windscreen modelled as a single surface, or two parallel faces 3–4 mm apart) does not appreciably displace the image: what is not reflected passes straight through. For a sheet with normal-incidence Fresnel reflectance $R=((\eta-1)/(\eta+1))^2$ at each face, summing the multiple reflections, the reflected fraction is $2R/(1+R)$; for $\eta=1.5$, $R=0.04$ and the sheet reflects $\approx 7.7\,\%$. athenea classifies automatically: a part is a sheet if it has open edges or if its mean thickness $2V/A$ (volume over area) is small compared with the cell.

**Roughness widened by the spread of normals (Toksvig).** A Gaussian has a single normal, but beneath it the original surface may curve; a sharp reflection that on the mesh moves along a chrome trim appears on the Gaussian as a blot the width of the Gaussian. Toksvig (2005, [20]) proposed treating the variation of normals under a texel as additional variance of the microfacet distribution. In athenea, with $\alpha=r^2$ ($r$ the perceptual roughness) and $\sigma_n$ the angular spread of normals under the Gaussian,
$$\alpha'^2 = \alpha^2 + 2\sigma_n^2 .$$

*References:* [20], [30] in §10; OpenPBR specification at academysoftwarefoundation.github.io/OpenPBR; code: `plugins/mesh2splat/mesh2splat.slang` (`m2sSpread`, `m2sWidened`, `m2sGlassCovers`), `apps/athenea/src/CmdMesh2Splat.cpp` (`classifySheets`, `thinGlassOf`), `shaders/athenea/usd/mesh2splat_sheet.slang`; `docs/decisions.md`, sections "A reflection off a curved strip is the mean over the gaussian (Toksvig)" and "A windscreen modelled as one surface is a sheet (`--thin-glass`)".
:::

::: {.nota}
**Spherical harmonics (SH).** The real spherical harmonics $Y_{lm}(\omega)$, $l\ge 0$, $-l\le m\le l$, are an orthonormal basis of functions on the sphere, analogous to the Fourier series on the circle: $l$ is the angular "frequency" (*band*). Truncating at degree $L$ gives $(L+1)^2$ functions: 1, 4, 9 or 16 for degrees 0 to 3. A function $f(\omega)$ is approximated by $f \approx \sum_{l\le L,m} f_{lm} Y_{lm}$ with $f_{lm} = \int f\,Y_{lm}\,d\omega$. The property that makes SH useful for lighting: the inner product of two functions becomes the dot product of their coefficient vectors, $\int f g\,d\omega \approx \sum_k f_k g_k$. Degree 3 represents variations of about $180°/3 \approx 60°$: enough for diffuse light and soft shadows, not for sharp reflections.

**Transfer of direct and indirect light (PRT).** This is the *precomputed radiance transfer* of Sloan, Kautz and Snyder ([14]). For a diffuse surface with albedo $\rho$ lit by a distant environment $L(\omega)$,
$$L_o = \frac{\rho}{\pi}\int_{\Omega} L(\omega)\,V(\omega)\,\max(\mathbf{n}\cdot\omega,0)\,d\omega \approx \frac{\rho}{\pi}\sum_k L_k\,T_k,$$
where $V$ is the visibility and $T_k = \int V(\omega)\max(\mathbf{n}\cdot\omega,0)\,Y_k(\omega)\,d\omega$ is the *transfer*: it does not depend on the sky and is precomputed once per point. The *direct* part is this one (what arrives from the sky without bouncing, one number per coefficient); the *indirect* part adds what arrives after bouncing off other parts of the asset, with their colour (three numbers per coefficient). In athenea, a degree-3 TX transfer is $16 + 48 + 48 = 112$ f16 values per Gaussian: 16 for the direct half (scalar), 48 for the indirect (rgb) and 48 for the reflected field (16 rgb coefficients). The bake does not project: it *fits* the coefficients by least squares over the hemisphere the surface faces.

**Reflected field.** What a reflection sees in the *closed* directions (occluded by the asset itself): the ground, the bodywork in the chrome, the dark interior behind a windscreen. It is stored normalised over those directions and coupled to the sky at draw time, so that it changes if the sky changes.

**256 bits of open directions.** An octahedral grid of $16\times16$ cells over the whole sphere; each bit says whether a ray in that direction escaped to the sky. A cell subtends about $11°$. At draw time, each specular lobe is read as a cone over these cells to know what fraction of its reflection sees the sky (and is therefore taken from the environment map) and what fraction is closed (and is taken from the reflected field).

**The material's marks.** Bits and bytes per Gaussian with the kind of lobes (specular, coat and sheen weights, IOR, metalness, the original roughness before widening) that the relight needs to evaluate the material.

*References:* [14], [17] in §10; Green, R., "Spherical Harmonic Lighting: The Gritty Details" (GDC 2003); code: `shaders/athenea/common/transfer_layout.slang` (counts 9/16/36/64/84/112), `shaders/athenea/common/sh.slang` (`shBasisValue`, `evaluateRest`), `shaders/athenea/common/bake_fit.slang` (`bakeFit`), `shaders/athenea/usd/mesh2splat_bake.slang` (`m2sTransferInto`), `modules/technique/include/athenea/technique/PathTracer.h` (`BakePoints`); `docs/decisions.md`, sections "A transfer right under any sky and any light (task TX)" and "The bake fits the harmonics, and no longer projects them".
:::

::: {.nota}
**Relighting at draw time, and sharp reflections turned per pixel.** At draw time, the diffuse part is a dot product of the transfer with the sky's 16 coefficients. The specular part does not fit in degree-3 SH; it is evaluated with the material's lobes against the prefiltered sky (an octahedral mip chain by roughness, *split-sum*). Since a Gaussian has a single normal, the reflection would be constant over its whole footprint; to reproduce how the reflection turns over a curved surface, the **shape operator** (or Weingarten map) $S$ is stored: the symmetric $2\times2$ matrix that gives the variation of the normal with position on the surface, $d\mathbf{n} = S\,d\mathbf{x}$. Its eigenvalues are the principal curvatures $\kappa_1,\kappa_2$. At each pixel the offset $d\mathbf{x}$ from the Gaussian's centre is known and the normal is turned by $S\,d\mathbf{x}$, so that the reflection moves within the Gaussian as it would on the mesh. Three numbers $(S_{uu}, S_{uv}, S_{vv})$ are stored in the Gaussian's principal axes.

*References:* do Carmo, *Differential Geometry of Curves and Surfaces* (1976), §3.2; Karis, "Real Shading in Unreal Engine 4" (SIGGRAPH 2013, split-sum); code: `plugins/mesh2splat/mesh2splat.slang` (`m2sShape`), `shaders/athenea/splat/splat_relight.slang` (`specularLobe`, `splatFieldAlong`, `splatLobeOpen`), `shaders/athenea/splat/splat_project.slang` (the per-pixel turn and the *fold*).
:::

::: {.nota}
**Chevrolet Corvette C7 of 192 meshes; PSNR.** The PSNR (*peak signal-to-noise ratio*) between an image $I$ and the reference $K$, with values in $[0,1]$, is
$$\mathrm{PSNR} = -10\log_{10}\mathrm{MSE}, \qquad \mathrm{MSE} = \frac{1}{3N}\sum_{p,c}\big(I_{p,c}-K_{p,c}\big)^2 ,$$
in decibels (dB). It is logarithmic: +1 dB is equivalent to multiplying the MSE by $10^{-0.1}\approx 0.79$, +3 dB to halving it, +10 dB to dividing it by 10. In athenea it is computed on the display-encoded (sRGB) image, which is how a person sees it.

**How to read 16.55 → 17.88 dB and 22.18 → 23.90 dB.** On the car crop, +1.33 dB is an MSE $10^{-0.133}\approx 0.74$ times the previous one: 26 % less squared error from doubling the bake resolution. On the whole frame, +1.72 dB is an MSE $\approx 0.67$ times the previous one. The frame's PSNR is higher than the crop's because the frame includes a lot of background (the environment sky, identical in both images) that dilutes the error: that is why the crop is the demanding metric. Values around 17–24 dB are low in absolute terms (a video codec typically gives 35–45 dB), but here an entirely different representation (relit Gaussians) is being compared with a path trace of the mesh, with sharp reflections and glass, and phase differences in small reflections penalise the MSE heavily. What matters is the relative improvement between steps and the per-class ratios (§1.6, §8).

**Close-up level at 5120 px, 74.5 million leaves in 24 trees.** A bake whose cell is derived from a camera 5120 pixels wide (four times 1280): each part has Gaussians the size of one pixel of that camera, for close views. *Leaves* are the baked Gaussians (the leaves of the merge trees) and *trees* the merge trees per group of parts. An intermediate measurement during development gave 69.5 M leaves, with one part and the occlusion still pending; the paper's figure is that of the finished level.

*References:* Wang, Bovik et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (2004) for SSIM and the limitations of PSNR; code: `plugins/measure/Measure.cpp` (the PSNR of the encoded image, `-10 log10(mse)`), `plugins/measure/measure.slang`, `apps/athenea/src/CmdCompare.cpp`; Section 7 (the Corvette on an iPhone) and Section 8 of the paper.
:::

::: {.nota}
**A USD and Hydra renderer; every step as a GPU compute kernel.** *Hydra* is OpenUSD's rendering architecture: it separates the scene (*scene delegate* / *scene index*, which reads USD) from the renderer (*render delegate*, `HdRenderDelegate`), which receives *rprims* (mesh, points, curves), *sprims* (cameras, lights, materials) and *bprims* (buffers) through `Sync` calls and draws in `HdRenderPass::Execute`. A *render delegate* is a plugin: the same `hdAthenea` delegate can be loaded from `usdview`, from Blender (which has its own Hydra) or from the engine. A *compute kernel* (*compute shader*) is a GPU program launched over a grid of threads without going through the graphics pipeline; here it is written in Slang ([44]) and compiled to Metal, Vulkan or CUDA. The rule "every step that touches data is a kernel" avoids CPU↔GPU copies and ensures that the baker scales with the GPU; the CPU only reads files and keeps the books.

*References:* [43], [44] in §10; Hydra documentation at openusd.org ("Hydra 2.0", "Render Delegate"); code: `modules/usd/src/RenderDelegate.h` (`HdAtheneaRenderDelegate`), `modules/gpu/src/ComputeKernel.cpp` (`ComputeKernel::dispatch`, parameters bound by name); the repository's `CLAUDE.md`, rule "No CPU arithmetic on data".
:::

::: {.nota}
**Rigged assets: skeleton, a Jacobian with the weights' gradients.** A *rig* is the structure that deforms a model: a skeleton of *joints* and, per vertex, weights $w_j$ that say how much each joint moves it (*linear blend skinning*, LBS): $\mathbf{p}' = \sum_j w_j X_j\,\mathbf{q}$. For a Gaussian it is not enough to move the centre: its covariance has to be transformed, $\Sigma' = J\,\Sigma\,J^{\mathsf T}$, with $J$ the Jacobian of the map. Since the weights vary with position, the full Jacobian is
$$J = L(M)\Big[\sum_j w_j A_j L(G) + \sum_j (X_j \mathbf{q})\,\nabla w_j^{\mathsf T}\Big],$$
and the second term —the one most implementations omit— stretches the Gaussian where two joints disagree (on the sparrow's wing, by between 15 and 50 % at the 90th and 99th percentiles).

**Zonal lobes that turn with a limb.** An SH transfer in world coordinates does not turn with the bone; turning it costs a $9\times9$ or $16\times16$ matrix per Gaussian. A *zonal lobe* is a function symmetric about an axis $\mathbf{a}$, $g(\omega) = \sum_l z_l k_l Y_{l0}$ in the axis frame; turning it is turning the axis. The axis and the coefficients $z_l$ that best approximate the transfer are fitted (Sloan 2008, [16]) in the Gaussian's own frame.

**Per-part visibility fields.** Each body part bakes, in its bind pose, a grid of probes with how much light it blocks in each direction; at draw time, a Gaussian's visibility is the product of what each part answers in its own current frame. No rays.

*References:* [16], [41], [42] in §10; Magnenat-Thalmann et al. (1988) and Kavan et al. (2008) on LBS; code: `shaders/athenea/scene/splat_skin.slang` (`splatSkin`), `modules/scene/src/SplatSkinner.cpp`, `plugins/splattransferzonal/splattransferzonal.slang`, `shaders/athenea/splat/splat_visibility.slang`, `modules/technique/src/SplatVisibility.cpp`; `docs/decisions.md`, sections "The whole Jacobian of a skinned gaussian", "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame" and "A cloud shadows itself by part: baked once, read every frame, no ray".
:::

::: {.nota}
**Levels of detail: merge trees, a monotone and bounded screen error, culls and pages.** *LOD* (*level of detail*) is drawing fewer primitives when the object appears small. A *merge tree* is a binary tree whose leaves are the original Gaussians and whose internal nodes are Gaussians that approximate their two children (pairwise merge, matching moments: $w = w_a + w_b$, $\boldsymbol\mu = (w_a\boldsymbol\mu_a + w_b\boldsymbol\mu_b)/w$, $\Sigma = \sum_i \tfrac{w_i}{w}(\Sigma_i + (\boldsymbol\mu_i-\boldsymbol\mu)(\boldsymbol\mu_i-\boldsymbol\mu)^{\mathsf T})$). Each node carries an error $\varepsilon$ in metres, forced to be **monotone** towards the root ($\varepsilon(n) \ge \varepsilon(\text{children})$), and a bounding sphere $\rho(n)$; its projected error is $P(n) = \varepsilon\,\text{ppu}/\max(d-\rho,\text{near})$. The *cut* for a view is the set of nodes with $P(n) \le \tau < P(\text{parent})$: monotonicity guarantees that one local test per node suffices, in parallel, without walking ancestors (Nanite's idea, [35]). *Back-face cull*: a node whose normal cone faces entirely backwards on a closed shell is not drawn. *Occlusion cull*: a node whose open-direction mask is closed in the camera's direction cannot be seen from outside. *Pages*: blocks of nodes ordered by error, to stream the coarse first.

**"At a cost of about 0.1 dB".** The image drawn with the cut loses about 0.1 dB of PSNR against drawing all the leaves: an MSE only 2.3 % higher ($10^{0.01}$); that is, the LOD is barely noticeable in the metric.

*References:* [2], [9], [34], [35] in §10; code: `shaders/athenea/lod/lod_pairs.slang` (`mergeCost`, `pairMerge`), `shaders/athenea/lod/lod_tree_cut.slang` (the monotone error and the sphere), `shaders/athenea/lod/lod_pages.slang`, `shaders/athenea/lod/occlusion_mask.slang`, `modules/usd/src/Decimate.cpp`; `docs/decisions.md`, sections "Simplify: fewer gaussians by pairs, and surfels" and "Decimation: the levels of detail, cut once by what a merge loses".
:::

::: {.nota}
**The transfer in its cloud's own frame; relMSE $9\times10^{-6}$.** The SH coefficients, the reflected field and the open cells are directions of the file in which the bake was made. If the scene rotates the cloud (a Y-up asset in a Z-up scene), the sky has to be expressed in the cloud's frame: the rotated coefficients are $c'_k = \int Y_k(\omega)\,L(R\,\omega)\,d\omega$, computed once per instance with a quadrature exact for products of degree 6 (four Gauss–Legendre nodes in $z$ times eight steps in $\phi$). The *relMSE* is the relative mean squared error on linear values, $\frac{1}{N}\sum (a-b)^2/(b^2+\epsilon)$ with $\epsilon = 10^{-2}$; $9\times10^{-6}$ corresponds to a typical relative error of 0.3 % (square root), and `docs/decisions.md` records 61 dB of PSNR in the same test: the pawn rotated by 60° is practically identical to the unrotated one. The control test (rotating the other way) gives 11 dB, which shows that the metric tells the frames apart.

*References:* Gauss–Legendre: Abramowitz and Stegun, *Handbook of Mathematical Functions*, §25.4; code: `shaders/athenea/splat/sky_turn.slang` (`skyTurn`), `shaders/athenea/splat/splat_relight.slang` (`relightCloudFrame`, `transferAlong`), `shaders/athenea/reference/image_compare_hdr.slang` (relMSE); `docs/decisions.md`, section "A transfer the stage turns, and the gaussians' parameters drawn (19 Oct)".
:::

::: {.nota}
**Every parameter as an image channel (AOV).** An *AOV* (*arbitrary output variable*) or image channel is an additional image the renderer produces alongside the colour: depth, normals, albedo, IDs. Here any parameter of the Gaussian —albedo, roughness (widened or the material's), metalness, transmission, IOR, normal, opacity, size, sky visibility, bounce, fraction of open cells, emission, reflected field, baked light— is composited through the camera as if it were its colour. It serves to debug the bake, for compositing, and as training data for a learned relight.

*References:* code: `shaders/athenea/splat/splat_relight.slang` (constants `kSplatViewAlbedo` … `kSplatViewMaterialRoughness`), `modules/render/src/TileRasterizer.cpp` (parameter `splatView`), `modules/mcp/src/Tools.cpp` (argument `splatView` of the `render` tool); USD setting `athenea:splatView` in `modules/usd/src/StageRenderer.cpp`.
:::

::: {.nota}
**The master: full-precision USD; OpenFX node; Cryptomatte; Blender add-on.** The *master* is the reference output of the bake, in the sense of an audio or video "master": the highest-quality copy from which the distribution versions are derived. It is a USDC (binary *crate* USD) with the cloud as a `ParticleField` and its attributes (transfer, cells, shape operator, lobes) unquantised. *OpenFX* (OFX) is the open standard for image-effect plugins used by Nuke, Natron, DaVinci Resolve and Flame; athenea's node is a *generator* that draws the cloud to the compositor's camera, in a separate process. The layers it delivers: depth (view z), world position (P) and *Cryptomatte* (Friedman and Jones, [47]): an ID channel by name hash with coverage by ranks, which lets the compositor pull mattes of any object with correct antialiasing and transparency. The Blender add-on registers `hdAthenea` as a render engine through Blender's own Hydra.

*References:* [43], [47] in §10; OpenFX specification at openeffects.org; Cryptomatte at github.com/Psyop/Cryptomatte; code: `integrations/ofx/src/AtheneaOfx.cpp`, `integrations/ofx/src/RenderServer.cpp`, `shaders/athenea/technique/crypto_ids.slang`, `modules/usd/src/Export.cpp` (`writeParticleFieldStage`), `integrations/blender/athenea_hydra/__init__.py`; `docs/decisions.md`, sections "The OpenFX plugin: athenea's gaussians in Nuke (18 Oct)", "Layers and the sky (19 Oct)" and "A pixel says which prims it saw: Cryptomatte".
:::

::: {.nota}
**Web format: clustered PCA, vector quantisation, a 40-byte record, WebGPU.** *Clustered PCA* (CPCA, Sloan et al. 2003, [15]) groups the transfer vectors with k-means and, within each cluster $c$, approximates each vector as the mean plus a combination of $M$ principal directions: $\mathbf{t} \approx \bar{\mathbf{t}}_c + \sum_{j=1}^{M} q_j\,\mathbf{d}_{c,j}$. With 112 values per Gaussian and a small $M$ (eight weights per half in the compact format), the memory per Gaussian drops by an order of magnitude and reconstruction is a matrix-vector product on the GPU. *Vector quantisation* (VQ) replaces each 256-bit mask with the index of the nearest entry of a *codebook* (learnt with k-means under Hamming distance). The 40-byte compact record groups position quantised in its chunk's box (3 × u16), albedo, opacity, log scale, packed quaternion, octahedral normal, curvature, material index, cluster, CPCA weights and shadow index, in byte planes (SoA) to compress better. *WebGPU* ([46]) is the modern graphics API of the web (successor to WebGL), with compute shaders, which allows the same kind of kernels to run in an iPhone's browser.

*References:* [11], [15], [23], [46] in §10; Gersho and Gray, *Vector Quantization and Signal Compression* (1992); the sparkwebGPU player (public demos at https://sparkwebgpu.lucab.co.uk) reads the compact record, reconstructs the CPCA transfer on the GPU and decodes the `.athc` v3 container; code in athenea: `shaders/athenea/lod/transfer_codec.slang`, `shaders/athenea/lod/shadow_vq.slang`, `modules/lod/src/Athc.cpp`.
:::

::: {.nota}
**Keywords.** A summary of the vocabulary the paper takes as known and which these notes define at its first appearance: *relight baking* (precomputing transport in order to relight), *precomputed radiance transfer* (transport in SH, [14]), *spherical harmonics* (the basis of functions on the sphere), *Hydra* (USD's rendering architecture), *GPU-only pipelines* (pipelines in which the data do not return to the CPU between steps), *compositing* (combining image layers in a compositor such as Nuke) and *image channels* (AOVs).
:::
