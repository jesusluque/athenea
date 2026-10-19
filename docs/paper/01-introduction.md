<!-- Copyright (c) 2026 jesus luque. -->

# 1. Introduction

## 1.1 The problem

Production 3D assets are authored as meshes: polygons, layered physically based materials, textures, lights and,
for characters, skeletons, all composed in USD. They are rendered faithfully by a path tracer, which is too slow for
interactive use, and interactively by a rasteriser, which needs the asset's full geometry and texture set resident and
still approximates its light transport. 3D Gaussian splatting (3DGS) offers a third route: a cloud of anisotropic
Gaussians, sorted and blended in screen space, draws photographic detail at interactive rates on modest hardware. A
Gaussian cloud, however, is normally *trained* from photographs of a real scene. It holds the light of that capture, it
cannot be relit, and it has no material, no skeleton and no relation to an authored asset.

::: {.nota}
**Mesh, layered physically based materials, textures, lights, skeletons.** A *mesh* is a polygonal surface: vertices, faces (triangles or polygons) and per-vertex or per-face attributes (*primvars* in USD: normals, UVs, colours). "Physically based" materials describe how a surface reflects and transmits light through a BSDF that conserves energy and obeys reciprocity; "layered" means that the BSDF is a stack (coat over specular over base). Textures modulate those parameters across the surface. In USD all of this is *composed*: a car may be a root file that references dozens of sub-assets, each with its material *variants*.

**BRDF and BSDF.** The BRDF $f_r(\omega_i,\omega_o)$ (*bidirectional reflectance distribution function*, Nicodemus 1977) gives the fraction of radiance arriving from $\omega_i$ that leaves towards $\omega_o$, in sr$^{-1}$: $dL_o = f_r\,L_i\cos\theta_i\,d\omega_i$. The BSDF (*scattering*) extends it to transmission (BRDF + BTDF). A Lambertian diffuse is $f_r=\rho/\pi$; a microfacet specular lobe (Cook–Torrance, GGX, [24]) is $f_r = \frac{D(\mathbf{h})\,F(\omega_o\cdot\mathbf{h})\,G(\omega_i,\omega_o)}{4\,|\mathbf{n}\cdot\omega_i|\,|\mathbf{n}\cdot\omega_o|}$, with $D$ the normal distribution, $F$ the Fresnel term and $G$ the shadowing-masking term.

**Why a path tracer is slow and a rasteriser approximates.** The path tracer estimates a recursive integral by Monte Carlo: the noise decreases as $1/\sqrt{N}$, and a clean frame of a car with glass and chrome needs hundreds or thousands of paths per pixel. A rasteriser evaluates per pixel only the visible surface with a local model (point lights, prefiltered environment maps, *shadow maps*, *screen-space* AO/reflections): it is fast, but it does not know global visibility, interreflections or what is seen through glass, and it needs all the geometry and all the textures in memory.

**Trained 3DGS.** In Kerbl et al. [1] the Gaussians' parameters are optimised by gradient descent with a photometric loss $\mathcal{L} = (1-\lambda)\,\mathcal{L}_1 + \lambda\,\mathcal{L}_{\text{D-SSIM}}$ ($\lambda=0.2$) against photographs from cameras calibrated by *structure from motion* (COLMAP), with densification (cloning and splitting) and pruning. The resulting colour is the observed radiance: it includes the light, the shadows and the reflections of the capture. Hence the cloud "holds the light of that capture": separating material and illumination from photographs (*inverse rendering*) is an ill-posed problem.

*References:* [1], [24] in §10; Nicodemus et al., "Geometrical Considerations and Nomenclature for Reflectance" (NBS 1977); Cook and Torrance (1982); Schönberger and Frahm, "Structure-from-Motion Revisited" (CVPR 2016, COLMAP); code: `shaders/athenea/material/microfacet.slang`, `shaders/athenea/material/lobes.slang`.
:::

This paper addresses the converse problem: a **relight baker** for Gaussian splatting. Given a photoreal USD asset
(the case followed throughout is a Chevrolet Corvette C7 of 192 meshes with OpenPBR car paint, chrome, glass, rubber
and lamps), the baker produces a Gaussian representation that

::: {.nota}
**The converse problem.** In classic 3DGS one goes from images to Gaussians by optimisation; here one goes from a complete 3D model (mesh + materials + lights) to Gaussians by *conversion* and *bake*, without optimisation. It is the "converse" of the usual flow in the literature, and it has a decisive advantage: the baker knows the exact geometry, the materials and the normals, so it does not have to estimate them.

**The Corvette C7 as a case study.** It gathers the hard cases of a relight: OpenPBR car paint (a metallic base with flakes under a dielectric coat, which gives a sharp reflection of the environment over a diffuse colour), chrome (a conductor of almost zero roughness: a curved mirror, the worst case for a low-frequency representation), glass (transmission, refraction and Fresnel reflection), rubber (dark, rough diffuse) and lamps (glass over reflectors, with emission). 192 meshes means 192 parts with different materials and scales, from one-metre panels to one-centimetre badges.

*References:* [30] in §10 (OpenPBR, coat/specular/base layers); `docs/decisions.md`, sections "A car in gaussians, and the two things it found" and "A car's paint is a Schlick metal, not a conductor".
:::

1. **relights**: each Gaussian carries the light transport baked from the asset, so that the cloud answers any sky and
   any light at draw time, not only the illumination it was made under;
2. is **faithful to a path trace** of the source mesh under that lighting, judged image against image and region by
   region;
3. keeps a rig where the asset has one, so that an animated asset still moves and its light turns with it;
4. scales from far views to close-ups through levels of detail.

::: {.nota}
**The four requirements.** (1) *Relights*: the cloud stores the transport $\mathcal{T}$ and not the radiance $L_o$, so that the frame applies $\mathcal{T}$ to the lighting of the moment (an HDR sky and analytic lights). (2) *Faithful to a path trace*, "image against image and region by region": the global metric (the frame's PSNR) can hide large local errors in small areas, so it is also measured by regions or classes (paint, chrome, glass, rubber) through ratios of means against the reference. (3) *Keeps the rig*: the cloud is stored in bind pose with its joints and weights, and is deformed at draw time by skinning, instead of storing one cloud per frame. (4) *Scales through levels of detail*: the same cloud serves in a wide shot (few large Gaussians) and in a close-up (many small ones).

**Drawing under "any sky and any light".** The sky is an HDR *environment map* (in USD a `DomeLight` with a lat-long texture); its 16 SH coefficients feed the diffuse part, and its prefiltered mip chain the specular part. Analytic lights (sun, spots) are treated as directions whose visibility is read from the open cells.

*References:* [14] in §10; code: `shaders/athenea/splat/splat_relight.slang` (`splatSunOpen`, `splatSunShare`, `environmentBrdfOf`), `modules/technique/src/Environment.cpp`; `docs/decisions.md`, "Step 4: lights that are not the sky".
:::

Neither of the obvious routes meets these. Training a cloud from path-traced views of the asset gives a fixed
illumination and spends hours of optimisation per asset. Rasterising the mesh with an approximate shading model gives
up the path trace's light transport. The route taken here converts the mesh directly into Gaussians and bakes the
asset's light transport into each Gaussian with the renderer's own GPU path tracer; the relit cloud is then compared
with the path trace of the mesh. Reducing, compressing and streaming the baked cloud to a WebGPU player on a phone is
a parallel application of the same data (§7).

::: {.nota}
**Why not train from path-traced views.** It is the "obvious" route: render hundreds of views of the asset with a path tracer and train a 3DGS on them. It has three problems: (a) the resulting cloud bakes *one* illumination, that of the render; (b) the optimisation costs hours per asset on a desktop GPU and produces densities that depend on the cameras' coverage; (c) the relation to the asset is lost: no material, no rig, no parts. The trained relightable variants (Relightable 3D Gaussians, Gao et al. ECCV 2024; GS-IR, Liang et al. CVPR 2024; GaussianShader, Jiang et al. CVPR 2024) estimate normals and a BRDF per Gaussian from images, with decomposition errors the baker avoids because it knows them.

**Why not rasterise the mesh.** A rasteriser with an approximate shading model (like a game engine's) can be fast, but its global illumination is a collection of approximations (screen-space AO, screen-space reflections, *probes*); the paper's reference is the path trace, and that route gives it up from the outset. It also requires all the geometry to be resident, which clashes with the goal of a phone.

**The route taken.** Direct conversion (deterministic, without optimisation, minutes instead of hours) plus a bake with the *same* path tracer that produces the reference. Making the bake a variant of the path tracer's kernel —the first intersection is read from a buffer of points instead of a camera— guarantees that the lights, materials and bounces are exactly those of the reference: there is no second integrator that could diverge.

*References:* Gao, J. et al., "Relightable 3D Gaussians" (ECCV 2024); Liang, Z. et al., "GS-IR" (CVPR 2024); Jiang, Y. et al., "GaussianShader" (CVPR 2024); code: `modules/technique/src/PathTracer.cpp` (`PathTracer::bake`), `shaders/athenea/common/bake_fit.slang`; `docs/decisions.md`, section "Path traced into gaussians: the bake".
:::

## 1.2 Goals and constraints

The central requirement is fidelity of the relit Gaussians to the path-traced reference. The design requirements
below hold throughout and shape almost every design decision in Sections 2–7.

::: {.nota}
**Design requirements as hard constraints.** The paper distinguishes between the *goal* (maximum fidelity to the path trace) and the *constraints* that bound how it is reached. "Where quality and frame rate conflict, quality takes precedence" means that a speed optimisation is accepted only if its measured loss against the path trace is negligible; the opposite is the usual criterion in game engines.
:::

| requirement | statement | where it acts |
|---|---|---|
| Fidelity to the path trace | The reference is the path trace of the source mesh under the same lighting. Where quality and frame rate conflict, quality takes precedence. | §3.10, §4.10, §8 |
| GPU-only arithmetic | No arithmetic on scene data runs on the CPU: no CPU renderer, fallback or test oracle. The host reads files and keeps the books (counts, slots, queues). | Every section; §2.1 states the rule |
| Pure 3DGS at draw time | What is drawn is Gaussians only. The ground is not a mesh: its interaction with the object is converted into Gaussians (a shadow catcher). A mesh is only an input to the bake. | §3.9, §7.9 |
| Automatic, not manual | Per-asset choices are detected from the data: the cell size from a camera, solid versus thin glass, closed shells, the ground for the shadow catcher. Options only override. | §3.3, §3.5, §3.8, §3.9 |
| Small pieces held to the same standard | Badges, grilles, trims and lamps are judged as carefully as panels, close up as well as far. | §3.11, §6.9, §7.12 |

::: {.nota}
**GPU-only arithmetic, and "CPU test oracle".** A *test oracle* is the mechanism that decides whether a test's result is correct. The usual practice in graphics is to compare with a reference implementation on the CPU; athenea forbids it: the references are kernels (for example, a `ReferenceRenderer` that draws Gaussians without tiles, or the evaluation of MaterialX by its own genglsl compiled into a kernel), and the CPU only reads counters or aggregate metrics (p99, maximum, number of pixels more than 2 codes apart). The rule has a practical reason (an asset of 70 M Gaussians does not fit in a reasonable CPU loop, and every GPU→CPU data trip is a synchronisation) and one of correctness: a single implementation of each formula, with no duplicates to diverge.

**Pure 3DGS at draw time; shadow catcher.** No primitive other than a Gaussian reaches the frame. The ground, which in a product shot is usually a plane with the object's shadow, is replaced by a *shadow catcher*: a layer of Gaussians on the ground plane that carries, in its bake, how much the object darkens it (the fraction of sky and light blocked) and is composited by multiplying what lies behind. "A mesh is only an input to the bake": meshes feed the conversion and the bake's path tracer, never the frame.

**Automatic, not manual.** The cell size comes from a camera; solid or sheet glass, from the topology and the mean thickness $2V/A$; *closed shells* (closed surfaces, without border edges, whose interior is never seen and whose back faces can be discarded), from counting edges on the device; the shadow catcher's ground, from the asset's box. The CLI options (`--thin-glass`, `--solid-glass`…) only override.

**Small pieces held to the same standard.** A global PSNR hardly moves if a 2 cm badge comes out wrong; that is why they are measured separately, and the LOD has guards that prevent merging across colour or material borders, and that prevent *needles* (very elongated merges on trims).

*References:* the repository's `CLAUDE.md`, "Rules this codebase holds to"; code: `modules/render/src/ReferenceRenderer.cpp` (`compareImages`), `modules/usd/src/ShadowCatcher.cpp`, `shaders/athenea/splat/splat_project.slang` (`splatProjectCatcher`, `catcherOpacity`), `apps/athenea/src/CmdMesh2Splat.cpp` (`classifySheets`), `shaders/athenea/lod/lod_pairs.slang` (`mergeCost`: material, colour, gloss and aspect guards); `docs/decisions.md`, sections "The ground's shadow as gaussians: a shadow catcher (task PLAY-G)" and "Opacity is coverage, not glass".
:::

## 1.3 Contributions

The contributions described in this paper are the following.

1. **A relightable transfer baked into each Gaussian** (TX, Section 4), the core of the baker: degree-3 spherical
   harmonics of direct and indirect transfer, a reflected field normalised over closed directions and coupled to the
   sky at draw time, 256 bits of open directions read as cones by each reflection lobe, and the material's marks;
   per-pixel sharp lobes driven by a stored shape operator, with the bake's widening undone and a tube distinguished
   from a fold.
2. **A mesh-to-Gaussian conversion** (mesh2splat, Section 3) that prepares the baker's input, run as an AOFX effect on
   the device, with cells derived from a camera's pixel footprint, opacity that reproduces a surface's coverage, hard
   edges and closed shells found on the device, Toksvig widening of roughness by the normals' spread, layered OpenPBR
   capture, and automatic glass classification.
3. **A GPU-only USD renderer** (athenea, Section 2) on which the baker runs, drawing meshes, points and Gaussian splats
   through Hydra, with a tile rasteriser and a Gaussian ray tracer that share one Gaussian layout, a MaterialX-to-Slang
   material compiler checked against MaterialX itself, a path tracer with multiple importance sampling used as ground
   truth and as the bake's integrator, and a verification discipline in which every image comparison is itself a GPU
   kernel.
4. **Skinned Gaussian clouds** (Section 5) that carry the rig rather than the frames: a skinning kernel using the full
   Jacobian including the weights' gradients, zonal transfer lobes that turn with the Gaussian, per-part visibility
   fields read as a product of table lookups, and decimation that keeps the rig.
5. **Per-part merge trees with a bounded error cut** (Section 6): pair merging under guards that respect parts and
   materials, a least-squares coverage opacity for merges, an error made monotone with covering spheres so that one
   parallel test selects the cut, back-face culling of closed shells, a conservative shared occlusion codebook, and
   pages ordered by the parent's error.
6. **A measured account** (Section 8) of what each step of the bake buys against the path trace, on one production
   asset, through the relit leaves, the cuts and the 640, 1280 and 5120 px bakes.
7. **A web format derived from the master** (Part II, Section 7): a `.athc` v3 container with byte-plane encodings, clustered PCA
   of the transfer kept compressed on the GPU, vector-quantised open-direction bits, a 40-byte compact record,
   quantised traversal records of 5–11 bytes, and a frame-time and still-frame policy for a WebGPU player on an
   iPhone.
8. **The baked cloud as image channels and in production hosts** (Sections 4.11 and 2.14): every parameter a Gaussian
   carries drawn through a camera as a channel of its own, the material and geometry beside the precomputed light
   transport, as the inputs and targets of a learned relighting; an OpenFX node that renders the relit cloud inside a
   compositor (Nuke, Natron) out of process, with depth, world position and Cryptomatte as planes and a sky the artist
   swaps; and a Blender add-on that draws it as a render engine.

::: {.nota}
**Contribution 1: the TX transfer.** A breakdown of its pieces.

*Degree 3, direct and indirect.* 16 coefficients for the light that arrives directly from the sky (one scalar per coefficient: the visibility times the cosine, with no colour because the colour is set by the albedo and the sky) and 48 for the light that arrives after bouncing off the asset itself (rgb, because the bounce is tinted by the colour of what it bounces off). Going from degree 2 (9) to 3 (16) improves contact shadows and the directionality of occlusion; `docs/decisions.md` records it as "Step 3".

*A reflected field normalised over the closed directions.* For a specular lobe pointing at occluded directions, what is reflected is not the sky but the object itself or the ground. The bake stores, in 16 rgb coefficients, the radiance of those closed directions divided by the closed fraction (normalised), so that at draw time it is multiplied by the closed fraction of the particular lobe and coupled to the sky (what is seen in the reflection also depends on the sky that lights it).

*256 bits of open directions read as cones.* A GGX lobe of roughness $\alpha$ has an angular width of order $\arctan\alpha$; the code takes a cone of half-angle $\min(1.4\arctan\alpha,\,1.2)$ around the reflected direction and counts what fraction of the octahedral cells under it are open. That fraction splits the lobe between the sky (open) and the reflected field (closed).

*Per-pixel sharp lobes driven by a shape operator, with the bake's widening undone.* The stored roughness is widened by Toksvig (by the curvature under the Gaussian). If the normal is also turned per pixel by the shape operator, the curvature would be counted twice; that is why the sharp lobes read the sky at the roughness *without* widening (`unwidened`). *A tube distinguished from a fold*: if the normal's turn over the footprint is moderate (a tubular trim, under about 0.7 rad) it is applied in full; if it is large (normals smoothed across a crease, over 1.5 rad) it is not real geometry but an interpolation artefact, and instead of turning, the lobe is widened.

*References:* [14], [18], [19] in §10; code: `shaders/athenea/splat/splat_relight.slang` (`splatFieldCoupling`, `splatFieldAlong`, `splatLobeOpen`, `splatConeOpen`, `splatCellsOpen`, `unwidened`), `shaders/athenea/splat/splat_project.slang` (`kFoldTurnLow`, `foldWiden`, `kCoatFoldRoughness`); `docs/decisions.md`, sections "Step 1: a reflection's occlusion has a direction", "Step 2: what the closed directions show", "Step 3: degree 3, and a maximum level to measure against" and "A TX transfer's reflection turns across the gaussian (proposals 044, 046, 049)".
:::

::: {.nota}
**Contribution 2: the mesh2splat conversion as an AOFX effect.** *AOFX* is aopenfx's plugin interface for GPU effects (the SDK and the *host* shared by the engine and the compositor): an effect is a *bundle* with a Slang kernel and a parameter descriptor, and the host runs it on device buffers. Making the conversion an effect means that the same conversion runs inside the engine, the CLI or a compositor, with no duplicated code.

*Cells from a pixel's footprint.* See the note on the abstract: $\text{cell} = z\,(\text{aperture}/\text{pixels})/\text{focal}$, with $z$ the closest distance from the camera to the part.

*Opacity that reproduces coverage.* The positive root of $\tfrac{\pi\sigma^2}{2}\alpha^2 + 2\pi\sigma^2\alpha = -\ln(1-F)$. Without the quadratic term, a surface with coverage 0.75 covered 0.77; with a naive opacity of 0.6 a glass covered 98 %.

*Hard edges and closed shells found on the device.* Edges are identified by the position of their vertices (not by index, because production meshes duplicate vertices at UV seams), sorted with a radix sort on the GPU and their repetitions counted: an edge with a single triangle is an open border; with two, the dihedral angle is examined to decide whether it is a crease; with more than two, it is non-manifold.

*Toksvig widening.* $\alpha'^2 = \alpha^2 + 2\sigma_n^2$, with $\sigma_n$ estimated as the maximum turn of the normal per unit length between the triangle's corners times the Gaussian's width.

*Layered OpenPBR capture and glass classification.* Per Gaussian, the weights and parameters of the specular, coat and sheen (including their textures) are read from the material; glass is classified by topology (open borders on more than 1/200 of the edges) and by thickness ($2V/A$ below four cells or 1/50 of the part's size).

*References:* [12], [13], [20], [30] in §10; aopenfx at github.com/jesusluque/aopenfx; code: `plugins/mesh2splat/Mesh2Splat.cpp`, `plugins/mesh2splat/mesh2splat.slang` (`m2sEmit`, `m2sCoverageAlpha`, `m2sHoldEdges`, `m2sSpread`, `m2sWidened`, `m2sLayersAt`), `shaders/athenea/usd/mesh2splat_edges.slang`, `shaders/athenea/usd/mesh2splat_sheet.slang`, `modules/aofx/host/src/Host.cpp`; `docs/decisions.md`, sections "mesh2splat: a mesh becomes a cloud, through the plugin interface", "One AOFX host, from aopenfx" and "A gaussian carries what its material layers over the base (proposal 026)".
:::

::: {.nota}
**Contribution 3: the renderer.** *Tile rasteriser*: the screen is divided into tiles of $16\times16$ pixels; each projected Gaussian is assigned to the tiles it touches, the (tile, depth) pairs are sorted with a radix sort on the GPU and each tile composites its Gaussians in order. *Gaussian ray tracer*: each Gaussian is wrapped in a stretched icosahedron (a proxy) inside a hardware or compute BVH; a ray gathers the Gaussians it crosses in segments, in order, and composites them (3DGRT, [7]). *One layout*: both read the same buffers (`positions`, `shape`, `sh`), with no conversion.

*A MaterialX-to-Slang compiler checked against MaterialX itself.* MaterialX ([31]) describes materials as node graphs; its code generator produces GLSL, MSL, OSL or Slang. athenea derives from MaterialX's `SlangShaderGenerator`, replacing the BSDF nodes with versions that stack lobes, and checks each graph by compiling at the same time a variant with the original genglsl responses and evaluating both in 65 536 directions: a maximum difference of $3.2\times10^{-5}$ over eleven graphs.

*A path tracer with multiple importance sampling (MIS).* When two sampling strategies can generate the same direction (sampling the light and sampling the BSDF), Veach and Guibas ([21]) combine them with weights; the power heuristic ($\beta=2$) is
$$w_s(x) = \frac{p_s(x)^2}{\sum_t p_t(x)^2},$$
which gives low variance both for small, bright lights (where light sampling wins) and for specular materials (where BSDF sampling wins).

*Verification as a kernel.* Every image comparison (difference histograms in 8-bit codes, relMSE, PSNR, SSIM) is a kernel; the CPU reads only the aggregate result.

*References:* [7], [21], [31], [44] in §10; code: `modules/render/src/TileRasterizer.cpp` (`kTile = 16`), `modules/render/src/GaussianRayTracer.cpp`, `shaders/athenea/rt/rt_proxy.slang`, `shaders/athenea/rt/rt_integrate.slang`, `modules/material/src/MaterialCompiler.cpp`, `tests/material/test_material_compiler.cpp`, `modules/technique/src/PathTracer.cpp` (parameter `mis`, "power heuristic"), `shaders/athenea/reference/image_compare.slang`; `docs/decisions.md`, sections "Ray tracing Gaussians", "MaterialX into Slang" and "Multiple importance sampling".
:::

::: {.nota}
**Contribution 4: skinned clouds that carry the rig rather than the frames.** Storing one cloud per animation frame multiplies the memory by the number of frames; storing the cloud in bind pose with four joints and weights per Gaussian (inherited from the triangle's vertices, interpolated barycentrically) and deforming it in a kernel costs one access per Gaussian per frame. The full Jacobian, the zonal lobes and the visibility fields are explained in the note on the abstract. *A product of table lookups*: a point's visibility is $V(\mathbf{x},\omega)\approx\prod_{p} V_p(T_p^{-1}\mathbf{x},\,T_p^{-1}\omega)$, one trilinear lookup per part $p$, instead of tracing rays. *Decimation that keeps the rig*: when Gaussians are merged, their joints and weights are merged too (by mass), so that a character's levels of detail still animate.

*References:* [41], [42] in §10; code: `shaders/athenea/scene/splat_skin.slang`, `modules/scene/src/SplatSkinner.cpp`, `shaders/athenea/splat/splat_visibility_read.slang`, `shaders/athenea/splat/splat_visibility_bake.slang`, `modules/usd/src/SkelSplat.cpp`; `docs/decisions.md`, sections "The file carries the rig, not the frames" and "A decimation keeps everything a cloud carries".
:::

::: {.nota}
**Contribution 5: per-part merge trees with a bounded error cut.** *Pair merging*: following NanoGS ([9]), each node looks among its neighbours in Morton order for the cheapest merge; two nodes that choose each other are merged. The cost adapts Runnalls' bound ([34]) on the Kullback–Leibler divergence lost by merging two components of a mixture,
$$B(i,j) = \tfrac12\big[\,w\log\det\Sigma_{ij} - w_i\log\det\Sigma_i - w_j\log\det\Sigma_j\,\big],$$
replacing the determinant (zero for a disc) with the covariance's second invariant $I_2$, the square of the disc's area. *Guards that respect parts and materials*: no merge across parts, of glass with non-glass, of a mirror with what is not one, of normals too far apart for the width of the highlight, across a colour border, or forming needles. *Least-squares coverage opacity*: the merge's opacity is fitted so that its footprint reproduces the composited coverage of its leaves. *An error made monotone with covering spheres*, *back-face culling*, a *conservative occlusion codebook* (an entry is the OR of its members' masks, so it never closes a direction that a member has open) and *pages ordered by the parent's error* (a node is needed when its parent's error exceeds the threshold, so that is the order in which they are needed).

*References:* [2], [9], [34], [35] in §10; Morton (1966) for Z-order; code: `shaders/athenea/lod/lod_pairs.slang` (`mergeCost`, `pairBest`, `pairMerge`), `shaders/athenea/lod/lod_tree_cut.slang`, `shaders/athenea/lod/lod_pages.slang` (`treeBandKeys`), `shaders/athenea/lod/occlusion_mask.slang` (`occlusionAssign`), `modules/usd/src/Decimate.cpp`, `apps/athenea/src/CmdDecimate.cpp`; `docs/decisions.md`, sections "A merge's second moment about its own centre (13 Oct)" and "What no camera sees, and merges no wider than the screen holds (9 Oct)".
:::

::: {.nota}
**Contributions 6 to 8.** *A measured account*: each step of the pipeline is evaluated with the same reference and the same metric, so that its contribution can be attributed. *A derived web format*: `.athc` v3 is sparkwebGPU's container; *byte-plane encodings* store each byte of an attribute in a separate plane (all the high bytes together, then the low ones), which groups similar values and improves gzip compression; *traversal records* of 5–11 bytes are the quantised records of the tree nodes the player walks to choose the cut; *still frame* is the policy of refining the image when the camera stops. *Channels and hosts*: the parameters as AOVs, the out-of-process OpenFX node (the engine runs in a separate render server so as not to load its Slang, its TBB and its USD inside Nuke) and the Blender add-on.

*References:* [11], [46], [47] in §10; code: `integrations/ofx/src/RenderServer.h` (why out of process); the `.athc` v3 reader and the traversal records of the sparkwebGPU player (public demos at https://sparkwebgpu.lucab.co.uk); `docs/decisions.md`, section "A transfer the stage turns, and the gaussians' parameters drawn (19 Oct)".
:::

## 1.4 Organisation

The work is two processes with different goals. **Part I** is the bake: it turns a production USD asset into a
relightable Gaussian cloud written as a USD file, the master, and is judged by fidelity to the path trace. **Part II**
derives from that master a web format for a phone, and is judged by size, memory and frame time, with its loss against
the master measured. Part I opens with what the master holds; Section 2 describes the engine and renderer on which the
baker runs, and the hosts that drive it. Section 3 describes the conversion from USD meshes to Gaussians, and Section 4
the light transport baked into them and how they are relit. Section 5 covers animated assets: skinning, the sparrow and
the pawn. Section 6 covers levels of detail. Part II opens with the derivation, parameter by parameter; Section 7
describes the web format and its WebGPU player. Section 8 consolidates the evaluation from Sections 2–7, Section 9 gathers
the limitations and the work left, and Section 10 lists the references, followed by an appendix on the repositories
and commands (Figure 1).

::: {.nota}
**Two processes, two criteria.** The separation is methodological: Part I is judged only by fidelity (with no size limit: the master may take gigabytes), Part II only by size, memory and frame time, with its loss measured *against the master* and not against the path trace. That way a compression error is not confused with a bake error, and the master can be reused for other targets (desktop, compositor) without re-encoding.

**Stage, USDC.** A *stage* is the composed view of a USD scene; USDC is the binary (*crate*) encoding of a layer, as opposed to USDA (text) and USDZ (zip).
:::

![Figure 1. The two processes. Part I, the bake: a USD stage is converted into Gaussians (§3), light transport is baked into each one by the path tracer (§4), animated assets keep their rig (§5) and merge trees give levels of detail (§6); the output is the master USDC, compared with a path trace of the source mesh. Part II, the web format: the master is read and encoded into a compact, streamed format for a WebGPU player on a phone (§7), measured as a loss against the master.](figures/fig-overview.svg)

::: {.nota}
**Reading Figure 1.** The flow of Part I: USD stage → mesh2splat (§3) → transfer bake with the path tracer (§4) → rig (§5) → merge trees (§6) → master USDC, compared with the path trace of the mesh. Part II reads the master and produces `.athc` for the WebGPU player (§7). Each part's comparison arrow points to a different reference: the path trace in Part I, the master in Part II.
:::

## 1.5 Related work

**Gaussian splatting.** Kerbl et al. (2023) introduced 3D Gaussian splatting: a scene as anisotropic Gaussians with
spherical-harmonic colour, optimised from photographs and drawn by a tile-based, depth-sorted rasteriser. Its
projection is the EWA splatting of Zwicker et al. (2001, 2002), a local affine approximation of the perspective map.
Mip-Splatting (Yu et al., 2024) adds a 3D smoothing filter and a 2D screen filter against aliasing; athenea's 0.3 px²
dilation with its opacity compensation is of that family (§2.5.1). 2D Gaussian splatting (Huang et al., 2024) flattens
each Gaussian to an oriented disc evaluated by ray–plane intersection, in the line of surfels (Pfister et al., 2000);
athenea's surfels follow it (§2.5.2). 3D Gaussian ray tracing (Moënne-Loccoz et al., 2024) traces particles through
bounding proxies in a BVH, which athenea's hardware route follows (§2.6).

::: {.nota}
**Kerbl et al. (2023).** The paper that popularised the field ([1]): anisotropic Gaussians with colour in degree-3 SH (48 rgb coefficients), optimised from photographs, drawn with a rasteriser of $16\times16$ tiles with a single global sort of (tile, depth) keys. It achieved quality comparable to Mip-NeRF 360 at over 100 fps at 1080p.

**EWA splatting (Zwicker et al.).** *Elliptical Weighted Average*: Heckbert (1989) proposed it for texture filtering; Zwicker carried it over to points and volumes. The perspective projection does not map a Gaussian to a Gaussian; the local affine approximation linearises it with the Jacobian at the centre, $J = \begin{pmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{pmatrix}$, and $\Sigma' = J W \Sigma W^{\mathsf T} J^{\mathsf T}$. The error grows towards the edges of the field of view, which is why athenea linearises at a point clamped to 1.3 times the frustum.

**Mip-Splatting and the 0.3 px² dilation.** A projected Gaussian smaller than a pixel produces aliasing. The original 3DGS adds $0.3$ to the diagonal of $\Sigma'$ (a screen filter) but without compensating, which fattens small Gaussians. Mip-Splatting ([4]) proposes a 3D filter (limited by the sampling frequency of the training cameras) and a 2D filter that approximates the pixel's box, and compensates the energy by multiplying the opacity by $\sqrt{\det\Sigma'/\det(\Sigma'+sI)}$. athenea uses $s=0.3$ with that compensation (gsplat's *antialiased* mode), plus a correction for surfels.

**2D Gaussian splatting and surfels.** 2DGS ([5]) uses discs (two scales) and evaluates each one by intersecting the pixel's ray with its plane, which gives exact depth and normals and avoids the affine approximation. Surfels (Pfister et al. [6]) are the classic version of oriented discs.

**3DGRT.** Moënne-Loccoz et al. ([7]) trace rays against the Gaussians using ray tracing hardware: they wrap each one in a stretched icosahedron, gather the $k$ closest hits in blocks and composite them in order. It enables secondary effects (reflections, refraction, distorted cameras) that the rasteriser does not support.

*References:* [1]–[7] in §10; Heckbert, "Fundamentals of Texture Mapping and Image Warping" (1989); gsplat (docs.gsplat.studio, `rasterize_mode="antialiased"`); code: `shaders/athenea/splat/splat_project.slang` (`a += 0.3`, `surfelPaid`), `shaders/athenea/splat/splat_blend.slang` (`surfelPower`), `shaders/athenea/rt/rt_proxy.slang`, `modules/render/src/GaussianRayTracer.cpp`.
:::

**From meshes to Gaussians.** Electronic Arts' mesh2splat converts a mesh into Gaussians by rasterising each triangle
in its dominant axis plane, one Gaussian per covered texel; athenea's conversion starts from it (§3.1). Mesh2GS
(arXiv 2606.21898) derives a Gaussian's size from a camera's pixel footprint, which athenea's `--cell-from-camera`
follows (§3.3).

::: {.nota}
**EA's mesh2splat (SEED).** It converts a mesh by rasterising each triangle in the plane of its normal's dominant axis (a triplanar projection): the triangle is projected onto the XY, YZ or XZ plane, whichever sees it best, rasterised at a fixed resolution, and each covered texel gives a Gaussian aligned with the triangle and coloured from its texture. athenea keeps that choice of axes (`projectionAxes`) and replaces the fixed resolution with the camera's cell.

**Mesh2GS.** It derives the Gaussian's size from the footprint of one pixel of a reference camera: a single rule for every part makes the density proportional to what the camera can resolve.

*References:* [12], [13] in §10; github.com/electronicarts/mesh2splat; code: `plugins/mesh2splat/mesh2splat.slang` (`projectionAxes`, `orthogonalUv`), `shaders/athenea/usd/mesh2splat_cells.slang`.
:::

**Precomputed light transport.** Precomputed radiance transfer (Sloan, Kautz and Snyder, 2002) stores, per point, the
projection of visibility times the cosine onto spherical harmonics, so that relighting is a dot product with the
environment's coefficients; Ramamoorthi and Hanrahan (2001) showed that nine coefficients represent irradiance well.
Clustered PCA (Sloan et al., 2003) compresses such transfers by clustering and a per-cluster principal basis, which
athenea uses both for the file and on the phone's GPU (§7.2). Zonal harmonics, which rotate by rotating an axis, are
fitted after Sloan (2008) (§4.4.5, §5.7). Filtered importance sampling (Křivánek and Colbert, 2008) prefilters the
environment (§4.3), and Lagarde and de Rousiers (2014) give the specular occlusion fit that the first transfer used
(§4.6.1).

::: {.nota}
**PRT and the dot product.** With the environment $L(\omega)=\sum_k L_k Y_k(\omega)$ and the transfer $T(\omega)=V(\omega)\max(\mathbf{n}\cdot\omega,0)$, the orthonormality of the basis turns the integral of the rendering equation into $\int L\,T\,d\omega = \sum_k L_k T_k$. For low-frequency environments (degree 2–4) the approximation is good; high frequencies (sharp shadows, reflections) require other bases (wavelets: Ng, Ramamoorthi and Hanrahan 2003) or, as in athenea, a separate treatment of the specular.

**Ramamoorthi and Hanrahan (2001).** The irradiance of an environment is its convolution with the clamped cosine, whose SH spectrum decays fast: $\hat A_0=\pi$, $\hat A_1 = 2\pi/3$, $\hat A_2=\pi/4$, $\hat A_3 = 0$, $\hat A_4 = -\pi/24$… Thus $E(\mathbf{n}) = \sum_{l,m}\hat A_l L_{lm} Y_{lm}(\mathbf{n})$ with nine coefficients has a mean error below 1 % ([17]).

**Clustered PCA (Sloan et al. 2003).** It combines VQ (k-means) and PCA: one cluster per region of the space of transfers, and within each a basis of a few principal directions. It also allows the relight to be evaluated per cluster: $\mathbf{L}\cdot\mathbf{t} = \mathbf{L}\cdot\bar{\mathbf{t}}_c + \sum_j q_j\,(\mathbf{L}\cdot\mathbf{d}_{c,j})$, so that the products $\mathbf{L}\cdot\mathbf{d}_{c,j}$ are computed once per cluster per frame ([15]).

**Zonal harmonics (Sloan 2008).** A function symmetric about the $z$ axis has only $m=0$ coefficients; rotated to an axis $\mathbf{a}$, its coefficients are $g_{lm} = z_l\,k_l\,Y_{lm}(\mathbf{a})$ with $k_l=\sqrt{4\pi/(2l+1)}$. Rotating it costs as much as rotating a vector ([16]).

**Filtered importance sampling (Křivánek and Colbert 2008).** To estimate an environment's specular integral with few samples, each sample reads the environment map at the mip level whose texel solid angle matches the solid angle the sample represents, $\Omega_s = 1/(N\,p(\omega))$: less noise in exchange for a little blur ([18]).

**Lagarde and de Rousiers (2014).** The Frostbite notes give a fit of *specular occlusion* from ambient occlusion and roughness, a single number for the whole hemisphere. athenea's first transfer used it; TX replaced it with the directional cells because "one number for the whole hemisphere" makes a door reflect open sky where its mirror image shows the ground ([19]).

*References:* [14]–[19] in §10; Ng, Ramamoorthi and Hanrahan, "All-Frequency Shadows Using Non-linear Wavelet Lighting Approximation" (SIGGRAPH 2003); code: `plugins/splattransferzonal/splattransferzonal.slang`, `shaders/athenea/lod/transfer_codec.slang`, `shaders/athenea/technique/env_prefilter.slang`; `docs/decisions.md`, section "A transfer right under any sky and any light (task TX)" (subsection "What was already there, and what was not").
:::

**Specular antialiasing.** Toksvig (2005) widens a microfacet distribution by the variance of the normals it
averages; athenea applies the same idea at conversion, to the spread of normals under one Gaussian (§3.7, §4.9).

::: {.nota}
**Toksvig (2005).** When a normal map is mipmapped, the averaged normals get shorter: $|\bar{\mathbf{n}}| < 1$ measures their spread. Toksvig turns that length into a lower Blinn–Phong exponent, $s' = \frac{|\bar{\mathbf{n}}|\,s}{|\bar{\mathbf{n}}| + s(1-|\bar{\mathbf{n}}|)}$, so that the distant surface shines less but wider, which is the correct mean of the highlight. The GGX version adds the variance of the slopes to $\alpha^2$ (Kaplanyan et al., "Filtering Distributions of Normals for Shading Antialiasing", HPG 2016). athenea applies the same idea to the geometric spread of normals under each Gaussian.

*References:* [20] in §10; Kaplanyan, Hill, Patney and Lefohn (HPG 2016); code: `plugins/mesh2splat/mesh2splat.slang` (`m2sSpread`, `m2sWidened`); `docs/decisions.md`, section "A reflection off a curved strip is the mean over the gaussian (Toksvig)".
:::

**Levels of detail for Gaussians.** Octree-GS (Ren et al., 2024) organises anchors in an octree selected by view
distance. Hierarchical 3D Gaussians (Kerbl et al., 2024) build a merge hierarchy from Gaussian moments and cut it by
screen size; athenea's original level of detail follows its moments (§6.1). NanoGS (Xiong et al., 2026) merges
mutual nearest neighbours pairwise, which athenea's per-part trees run on the device (§6.2). The cost uses Runnalls'
(2007) bound for Gaussian mixture reduction, and the error-band pages follow Nanite's (Karis et al., 2021) use of a
monotone error for a parallel cut (§6.3, §6.8).

::: {.nota}
**Octree-GS.** It organises *anchors* (Scaffold-GS: each anchor generates Gaussians with an MLP) in an octree and chooses the level by distance to the camera; discrete level-by-level LOD.

**Hierarchical 3D Gaussians (Kerbl et al. 2024).** They build a hierarchy by merging Gaussians by moments (mass, mean, covariance) and cut it by projected size on screen, interpolating between parent and children to avoid *popping*. Designed for scenes kilometres across.

**NanoGS.** Simplification by pairs of mutual nearest neighbours: a highly parallel scheme (all nodes choose at once, reciprocal pairs are merged) that suits the GPU.

**Runnalls (2007).** Gaussian mixture reduction in target tracking: the upper bound on the KL divergence between the mixture before and after merging two components is the formula $B(i,j)$ of the note on contribution 5; it is cheap and requires no integration.

**Nanite.** Unreal Engine 5's virtualised geometry system builds a DAG of clusters of simplified triangles and assigns each group a monotone error (a parent's never smaller than its children's) with a bounding sphere that is also monotone; that way the decision "do I draw this cluster?" is taken in parallel per cluster without walking the graph, and two neighbouring clusters never take incompatible decisions.

*References:* [2], [8], [9], [34], [35] in §10; Lu et al., "Scaffold-GS" (CVPR 2024); code: `shaders/athenea/lod/lod_leaf_moments.slang`, `shaders/athenea/lod/lod_merge_moments.slang`, `shaders/athenea/lod/lod_pairs.slang`, `shaders/athenea/lod/lod_tree_cut.slang`; `docs/decisions.md`, sections "Levels of detail" and "Decimation: the levels of detail, cut once by what a merge loses".
:::

**Engines and formats.** The engine is built on OpenUSD and Hydra, MaterialX and OpenPBR materials, Slang shaders
compiled through slang-rhi to Metal, Vulkan and CUDA, Intel Open Image Denoise and OpenColorIO. The player is a fork
of Spark with a WebGPU backend.

::: {.nota}
**The engine's stack.** *OpenUSD and Hydra* ([43]) for the scene and the rendering architecture. *MaterialX* ([31]) and *OpenPBR* ([30]) for the materials. *Slang* ([44], He, Fatahalian and Foley 2018) is a shading language derived from HLSL with modules, generics and interfaces, and with reflection that allows parameters to be bound by name; it compiles to SPIR-V (Vulkan), MSL (Metal), PTX (CUDA), WGSL (WebGPU) and others, and is now maintained by Khronos. *slang-rhi* is its hardware abstraction layer (*render hardware interface*): it creates the device, buffers, pipelines and queues on Metal, Vulkan, D3D12 or CUDA with a single API. *Intel Open Image Denoise* ([45]) is a neural-network denoiser for path tracing that uses albedo and normal as guides. *OpenColorIO* ([40]) is the industry's colour management library (view transforms, ACES); athenea compiles its transforms into a kernel. *Spark* ([11]) is a 3DGS renderer for three.js by World Labs; athenea's player is a fork with a WebGPU backend.

*References:* [11], [30], [31], [40], [43]–[46] in §10; shader-slang.org (Slang documentation and user guide); github.com/shader-slang/slang-rhi; code: `modules/gpu/src/Device.cpp`, `modules/gpu/src/ShaderLibrary.cpp`, `modules/technique/src/Denoiser.cpp`, `modules/technique/src/Ocio.cpp`, `modules/colour/src/ColourCompiler.cpp`; `docs/decisions.md`, sections "Complete USD: toolchain (M0)" and "Colour: OpenColorIO as a compiler, and a texture read in its own colour space".
:::

## 1.6 Conventions

Unless stated otherwise, measurements were taken on an Apple M5 Pro under Metal. Images are compared on the GPU
(§2.15). **PSNR** is in dB on display-encoded images; **relMSE** is relative mean squared error on linear values.
**The path trace** (PT) of the Corvette is the mesh path-traced from `/World/Camera` at 1920 × 1080 under the
`autoshop_01_4k` sky. **Car crop** is PSNR on the 900 × 420 crop of the car, **frame** on the whole frame; a **class
ratio** is a region's mean over the path trace's. The validate-corvette harness of the player measures other regions
(car, door, sill, bumper, hood), which are not the car crop (§7.12.3). The words *Gaussian* and *splat* are used
interchangeably; a *leaf* is a baked Gaussian, a *merge* a node of a merge tree, and a *cut* the set of nodes drawn for
one view. *Transfer* is the baked light transport (§4), *TX* its current form.

::: {.nota}
**Apple M5 Pro under Metal.** An integrated GPU with unified memory (CPU and GPU share the same physical memory), which makes copies cheap but makes GPU memory compete with system memory; Metal is Apple's graphics API, to which slang-rhi compiles via MSL.

**PSNR on display-encoded images.** The renderer's images are linear (proportional to radiance). Before the PSNR is measured they are encoded with the sRGB OETF, $V = 12.92\,L$ if $L\le 0.0031308$ and $V = 1.055\,L^{1/2.4} - 0.055$ otherwise, after clamping to $[0,1]$. That way an error in the shadows weighs more than one in the highlights, as the eye perceives it, and the PSNR is comparable with that of the 3DGS literature, which measures on 8-bit images.

**relMSE on linear values.** $\mathrm{relMSE} = \frac{1}{N}\sum_{p}\frac{1}{3}\sum_c \frac{(a_{p,c}-b_{p,c})^2}{b_{p,c}^2 + 10^{-2}}$, with $b$ the reference. It is the standard metric for comparing HDR renders (from the denoising literature: Rousselle et al. 2011): the error is measured relative to the magnitude of the reference, and the $10^{-2}$ avoids dividing by zero in the blacks. A relMSE of $10^{-2}$ corresponds to a typical relative error of 10 %; $10^{-4}$, of 1 %.

**The Corvette's path trace.** The stage's `/World/Camera` camera, $1920\times1080$, lit by the HDRI `autoshop_01_4k` (from Poly Haven, an indoor workshop with windows and fluorescent tubes: an environment with compact, bright sources that produce sharp reflections on the paint, the demanding case). *Car crop*: $900\times420$ pixels around the car. *Class ratio*: $\bar I_{\text{region}}/\bar K_{\text{region}}$; 1 is exact, 0.72 is 28 % darker. It measures brightness bias (which PSNR mixes with noise and shape differences), and that is why it is used to judge small regions or glass.

**Leaf, merge, cut, transfer, TX.** *Leaf*: a baked Gaussian, a leaf of a merge tree. *Merge*: an internal node. *Cut*: a set of nodes that covers each leaf exactly once (each leaf is drawn by itself or by a single ancestor). *Transfer*: the baked transport; *TX* its current form (degree 3, reflected field, 256 cells), as opposed to the "first transfer" (degree 2, 64 bits over the hemisphere, Lagarde's specular occlusion).

*References:* IEC 61966-2-1 (sRGB); Rousselle, Knaus and Zwicker, "Adaptive Sampling and Reconstruction using Greedy Error Minimization" (SIGGRAPH Asia 2011); polyhaven.com/a/autoshop_01; code: `shaders/athenea/reference/image_compare.slang` (`encode`), `shaders/athenea/reference/image_compare_hdr.slang` (relMSE with $10^{-2}$), `plugins/measure/Measure.cpp`; the validate-corvette harness of the sparkwebGPU player; `docs/decisions.md`, section "Measure: `athenea compare` as an AOFX effect".
:::
