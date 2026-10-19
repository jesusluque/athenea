<!-- Copyright (c) 2026 jesus luque. -->

# Part I. The bake: from production USD to a relightable master USDC {.unnumbered}

The first process turns a production USD asset into a Gaussian cloud that can be relit, and writes it as a USD file:
the **master**. Its goal is fidelity. The master is judged by one criterion, its relit image against a path trace of
the source mesh under the same sky and lights, and every choice in it -- the cell size, the bake's samples, the bits
per direction, the merge cost -- is made by that measurement, not by its size.

**Input.** A USD stage as production delivers it: meshes with their MaterialX or OpenPBR materials, UsdLux lights and
domes, cameras, and for an animated asset its skeleton and clips.

**The process**, in the order it runs, each step a set of GPU kernels on the engine of Section 2:

1. **Conversion** (Section 3). Each mesh becomes anisotropic Gaussians, one per cell of a lattice sized from a camera's
   pixel footprint, with the coverage of the surface as opacity, edges held where it stops or folds, the layered
   material captured per Gaussian, glass classified as solid or thin wall, and roughness widened by the normals' spread.
2. **The transfer bake** (Section 4). The engine's path tracer traces each Gaussian's hemisphere and stores how any sky
   reaches it: degree-3 harmonics of direct and indirect light, a reflected field, 256 bits of open directions and the
   surface's curvature. With these the cloud is relit at draw time under skies and lights it was never baked under.
3. **The rig and visibility** of an animated asset (Section 5): influences and their gradients per Gaussian, zonal
   transfer lobes that turn with a limb, and per-part visibility fields.
4. **Levels of detail** (Section 6): per-part merge trees whose cut by screen error is bounded, back-face and shared
   occlusion culls, and pages ordered by error, written beside the leaves in the same file format.

**Output: the master USDC.** A `ParticleField3DGaussianSplat` per cloud (or per merge tree), whose standard attributes
(positions, orientations, scales, opacities, radiance harmonics) are read by any USD application, and whose athenea
primvars carry the rest at full precision:

| parameter | primvar (`primvars:athenea:splat:`) | per Gaussian | § |
|---|---|---|---|
| material | `normal`, `metallic`, `roughness`, `materialRoughness`, `transmission`, `specularWeight`, `coatWeight`, `sheenColor`, `emission`, `schlickMetal`, `thinWalled`, `closed`; `ior` per cloud | a value per property | 3.6–3.8, 6.6 |
| transfer (TX) | `transferDirect` (16), `transferIndirect` (48 rgb), `transferReflected` (48 rgb) | 112 values | 4.2, 4.4 |
| open directions | `shadowBits` | 256 bits | 4.4.2 |
| curvature | `curvature` | $(S_{uu}, S_{uv}, S_{vv})$, 3 floats | 4.8 |
| transfer and visibility, together | | 480 B in the file, 256 B on the device | 4.2 |
| rig | `skel:jointIndices`, `skel:jointWeights` (4 each), `jointWeightGradients` (half, 6), `transferZonal` (10), `skinningXforms` per joint | | 5.3, 5.5, 5.7 |
| levels of detail | `lodTree`, `lodParent`, `lodStart`, `lodSize`, `lodCost`, `lodSpill`, `coverage`, `coverageMass`, `occlusionIndex`; `occlusionCodebook` per tree; the merges written after the leaves | per node | 6.2–6.8 |
| ids | `cryptoObject`, the Cryptomatte id of the Gaussian's prim; `cryptoManifest` per cloud | 1 word | 2.5.2, 3.10 |

Sizes measured on the assets of this paper: the pawn's two clouds 489 MB; the sparrow, skinned with transfer and
visibility, 216 MB; the Corvette's relightable cloud of 3 M Gaussians 2.9 GB; the Corvette's close-up level as 24 merge
trees of 74.5 M leaves and 148.9 M nodes, 87.4 GB.

**Who reads the master.** The engine itself, through Hydra, on both its raster and its path tracer; the comparison
against the path trace (Section 8); the OpenFX node in a compositor and the Blender add-on (§2.14); the image channels
of §4.11. And every derived format, of which Part II describes one. The master is the source: a derived format is made
from it and cannot be turned back into it.

# 2. The athenea engine and renderer

athenea is a USD renderer built so that every step that touches scene data runs on the GPU. It draws three kinds of
primitive: meshes, with MaterialX materials and UsdLux lights; points; and 3D Gaussian splats. It offers two
techniques, an interactive raster and a path tracer that serves as ground truth. Hydra feeds both through the
`hdAthenea` render delegate. This section covers the engine's architecture, the algorithms each subsystem runs, how
correctness is established without a CPU oracle, and what has been measured. Unless stated otherwise, every number is
quoted from `docs/decisions.md`, with its unit and conditions, and was measured on an Apple M5 Pro under Metal.

## 2.1 Design rules

The codebase holds to a small number of rules. Where possible, each one is enforced by a test.

1. **No CPU arithmetic on data.** athenea has no CPU reference renderer, no CPU fallback and no CPU test oracle. The
   host reads files, parses headers, decompresses, and keeps the books: counts, slots and queues. Decoding, sorting,
   merging, culling, mip generation, light power prefixes and image comparison are all compute kernels. Steps that
   had run on the host were moved to the device; examples are per-element
   interleaving of `ParticleField` arrays, half-to-float conversion, per-pixel conversion of Hydra render buffers, and
   each light's cumulative share of the frame's power (`light_prefix.slang`). The one documented exception is Dear
   ImGui's tessellation, which processes viewer chrome and no scene data.
2. **Binding by name.** Every shader parameter is bound through reflection (`cursor["name"].setBinding(...)`,
   `ComputeKernel::dispatch`) and never by slot, which removes the class of errors in which a slot count
   drifts from the shader's declaration.
3. **Errors as values.** Errors are reported as `Result<T>` with `ATHENEA_TRY`, and nothing throws. OS calls
   live only in `modules/core/Platform`, which is also where a Windows port would start.
4. **One Slang, one slang-rhi, one TBB per process.** OpenUSD, OIDN and the engine share one TBB, and the
   `single_tbb` test checks the count. For the same reason OIDN is built from source: its release binaries ship a
   second `libtbb.12`.
5. **GPU devices only, everywhere.** OIDN is built for GPU devices only, because a denoiser that falls back to the
   CPU is a CPU fallback. A test whose capability is missing on a device skips and names the missing capability.

The backends are those of slang-rhi: Metal (macOS, iOS), Vulkan and CUDA (Linux, NVIDIA L4). All kernels are
written in Slang.

## 2.2 Module architecture

Each module under `modules/<name>` is a static library, `athenea::<name>`. Modules are listed in dependency order and
each links only the ones above it (`modules/CMakeLists.txt`) (Figure 2).

| Module | Contents |
|---|---|
| core | `Result`, logging, `Platform` (all OS calls) |
| ui | viewer panels as a description (`Controls.h`, `GaussianPanel.h`), walked by `athenea view` and the iOS app |
| sched | `FrameClock`: genlock PTP and ST 2059-1 alignment, timecode |
| image | `Image`: float32 RGBA premultiplied, bottom-left origin; the type gpe and AOFX exchange |
| io | CPU file readers (PLY header, SPZ, SOG zip/WebP, IES), EXR with attributes |
| gpu | slang-rhi device, `ShaderLibrary`, `ComputeKernel`, `CommandBatch`, `Buffer`, `AsyncReadback`; `gpu/algo` (PrefixSum, RadixSort) |
| gpu_host | gpe adopting slang-rhi's device: one `MTLDevice` or CUDA context, buffers shared without copies |
| colour | `ColourCompiler` (OpenColorIO as a compiler into Slang), `ColourNames` |
| scene | `CloudLoader`: raw records uploaded and decoded on the GPU into `GpuSplats` / `GpuPoints` |
| render | `TileRasterizer`, `GaussianRayTracer`, `PointRasterizer`, `ReferenceRenderer`, `Camera`/`Projection`, `SplatEdit` |
| geom | `MeshBuilder`: Hydra meshes triangulated, smooth-normalled and primvar-expanded on the GPU |
| material | `TextureStore`, `MaterialCompiler` (MaterialX → Slang), the lobe library |
| light | UsdLux lights on the device: one record per light, sampling, densities, light BVH |
| world | `GpuScene`, `Instancing`, `RayTracingScene` (BLAS/TLAS), `BvhScene` (two-level compute LBVH) |
| technique | `VisibilityRaster/Trace/Bvh`, shading, `AovShading`, path tracer, `Denoiser`, `DisplayTransform`, `SplatVisibility` |
| lod | `LodBuilder`, `CutSelector`, the `.athc` reader/writer and `StreamingPool` (Section 6) |
| usd | `Engine`, `StageRenderer`, `Export`, the `hdAthenea` plugin, codeless schemas |
| mcp | the engine as an MCP server (JSON-RPC 2.0) |
| aofx | this engine's side of the AOFX host (aopenfx `host/`), `renderEffect`, the bundles |
| view | `athenea view`: GLFW window, Dear ImGui on the engine's device |

![Figure 2. Engine modules in dependency order, with their main classes. Each static library links only the modules below it.](figures/fig-modules.svg)

The shader tree `shaders/athenea/` mirrors the modules (common, algo, scene, splat, rt, points, reference, lod, geom,
world, material, light, technique, usd, volume, view). Shaders are copied into the build tree and compiled at run
time.

## 2.3 GPU foundations

**Device and kernels.** `gpu::Device` wraps a slang-rhi device. `ComputeKernel` is one compute entry point compiled
as a pipeline. It is dispatched in threads, not groups: the group count is rounded up on every backend and each kernel
does its own bounds check. `RasterKernel` is a vertex and fragment pipeline that pulls its data from
`StructuredBuffer`s, with no vertex buffers or input layouts. `RayTracingKernel` covers OptiX and Vulkan RT
pipelines. On Metal, slang-rhi offers no ray tracing pipelines, so tracing uses inline `RayQuery` inside compute
kernels.

**ShaderLibrary.** The library loads Slang modules. A module may declare link-time constants
(`extern static const uint kName;`) and is then linked against a generated exports module, which yields one program
per distinct set of values. It also compiles modules that exist only as generated source, such as materials and the
colour functions; loading an existing name with different source is refused. A persistent `DiskShaderCache` writes
one file per compiled program, keyed by slang-rhi's hash of the linked program. With a warm cache the whole test
suite went from 89 s to 26 s.

**Parallel primitives.** `PrefixSum` and `RadixSort` are the only sorting and scanning in the engine. The tiled
radix route (`radix_block.slang`) is a reduce-then-scan form of the stable LSD pass. A group of 256 threads counts a
tile of 1024 keys in group memory, a digit-major prefix sum places each tile's share of each digit, and the group
sorts its tile with eight one-bit splits. It sorts 10 M 32-bit pairs in **12.4 ms**, against 39.9 ms for the chunked
passes it replaced from 65 536 elements up. It uses no wave operations and no device atomics.

**Mips.** slang-rhi does not generate mip chains, so `MipGenerator` is a kernel. Each texel of a level is the
area-weighted mean of the texels above it. An odd edge $2n+1$ folds into $n$ with weights $(n-x,\,n,\,x+1)/(2n+1)$,
so every source texel gives exactly $n/(2n+1)$ of itself to the level below and the chain preserves level 0's mean.
On sizes 64², 37×23, 1×9 and 128×5, the means agree to 2·10⁻⁶.

**Memory as a first-class error.** A Metal command buffer that runs out of memory used to assert inside slang-rhi on
Metal's completion thread, which ended the process. A patch now records the error on the queue
(`SLANG_E_OUT_OF_MEMORY`) and returns it from the next `submit()` or `waitOnHost()`. `Device::admit` refuses any
allocation that would exceed the budget: `ATHENEA_GPU_BUDGET`, or else Metal's `recommendedMaxWorkingSetSize`, which
is 18 186 MiB on the M5 Pro. An allocation must also fit the free physical memory less 1.5 GiB. On unified memory,
several processes each within Metal's figure can together exhaust physical memory and force the system to swap.
When a frame runs out of memory, `Engine::relieveMemory` steps down: it drops splat shadows first, then moves up to four LOD levels coarser, and finally reopens streams at half their budget.

## 2.4 The Gaussian representation

Every renderer, the loaders and the LOD share one GPU layout, `GpuSplats`. Each splat takes **32 bytes before its
harmonics**, defined in `common/packing.slang`:

| Field | Encoding | Reason |
|---|---|---|
| position, opacity | 4 × float32 | fp16 positions would step two pixels at 4K close up |
| rotation | smallest-three quaternion, 3 × 10 bits + dropped index | within a tenth of a degree |
| scale | f16 of $\log s$ per axis | 0.1 % relative precision at every size |
| base colour | 3 × f16 | may leave [0, 1] |
| SH rest | `shWords` × u32, f16 per coefficient | signed |
| normal (optional) | octahedral 2 × 16 bits | shading normal kept apart from the frame |
| emission (optional) | RGB9E5 | radiance the surface emits |

`CloudLoader` reads PLY, SPZ, SOG, `.splat` and USD `ParticleField` prims. It uploads the raw records and decodes
them on the GPU. For SOG, the host's only work is to unzip the archive and decode the WebP images.

**Linear light.** Every Gaussian is blended in linear Rec.709, and sRGB appears only where an image is displayed. A
capture keeps the sRGB harmonics it was trained with: each projection evaluates them, clamps at zero and converts to
linear (`cloudLight`) before blending. A cloud written by athenea declares that it holds light
(`primvars:athenea:splat:linear`) and is used as stored. On two half-opaque overlapping discs, the rasteriser, the
reference and the ray tracer each match the analytic over-operator at p99 0. Under the previous encoded blend, all
three read p99 26.

## 2.5 The tile rasteriser

`render::TileRasterizer` is the interactive route for splats. It is a sort-based, deterministic tile rasteriser in
the 3DGS family. Seven stages run on the device (Figure 3):

| Stage | Work |
|---|---|
| project | per splat: covariance, conic, colour, 24-bit depth key, footprint, exact count of tiles touched |
| sort | splats by depth (RadixSort) |
| gather | tiles-touched counts in depth order, prefix sum |
| emit | one (tile, splat) pair per touched tile, in depth order |
| sort | pairs by tile, stably, on tile bits only |
| ranges | start and end of each tile's pairs |
| blend | per pixel: its tile's pairs, front to back |

The engine sorts splats by depth once (N 32-bit keys) and then sorts pairs stably by tile (P keys of about 15 bits).
This gives every tile its splats in depth order, with ties broken by splat index, exactly as a single 64-bit
(tile, depth) sort would. The construction is deterministic. Tiles are 16 × 16 pixels.

![Figure 3. The tile rasterizer: seven compute stages, three per splat and four per (tile, splat) pair.](figures/fig-raster.svg)

### 2.5.1 Projection

A splat with rotation $R$ and scales $S=\mathrm{diag}(s)$ has the 3D covariance $\Sigma = R S S^\top R^\top$, built
in the cloud's own space. With $W$ the linear part of object-to-view and $J$ the Jacobian of the perspective map at
the view-space centre $t$, the screen covariance is the EWA local affine approximation

$$\Sigma_{2D} = (J W M)(J W M)^\top,\qquad M = R S,\qquad
J=\begin{pmatrix} f_x/t_z & 0 & -f_x t_x/t_z^2\\ 0 & f_y/t_z & -f_y t_y/t_z^2\end{pmatrix}.$$

Before $J$ is built, $t_x/t_z$ and $t_y/t_z$ are clamped to the frustum widened to $[-0.15, 1.15]$ of the image. A
splat far off screen would otherwise smear across the whole frame, and 3DGS clamps in the same way. Orthographic
cameras use $J=\mathrm{diag}(f_x,f_y)$.

**Shutter and lens, as convolutions.** A Gaussian smeared along a straight screen path $d$ under a uniform shutter
is that Gaussian convolved with a segment. Matching second moments turns this into a rank-one update. A thin lens
convolves with the disc of the circle of confusion:

$$\Sigma' = \Sigma + \tfrac{1}{12} d\,d^\top,\qquad
\Sigma' = \Sigma + \tfrac{R^2}{4} I,\quad R = f\,A\,\left|\tfrac1z - \tfrac1{z_f}\right|.$$

The displacement $d$ has two sources: what a skeleton moved the splat over the shutter (a per-splat buffer), and a
view-space step that combines camera and prim motion (`viewStep`). Both are projected by the same rows $JW$ as the
covariance. The update is checked against the same rasteriser run at N instants and averaged on the device
(§2.5.3).

**Antialiasing filter and the opacity it pays back.** The covariance is then dilated by 0.3 px², as in
Mip-Splatting's 2D filter. Each convolution spreads the splat, so its peak must drop by
$k = \sqrt{\det\Sigma_{\text{still}}/\det\Sigma'}$ to keep its energy. One square root covers the filter, the shutter
and the lens together. Scaling alone loses mass, however, because a splat is drawn only out to where it falls below
1/255, and the spread enlarged that area by $1/k$. A converted plane meant to cover 0.25 covered 0.18 at a twentieth
of the screen. The opacity is therefore paid back as

$$\alpha' = \begin{cases} k\alpha + (1-k)/255 & k\alpha \ge 1/255\\ k\alpha & \text{otherwise.}\end{cases}$$

This keeps the sharp splat's covered area (`paidOpacity`). Paying back every splat would also draw those the filter
had made too faint to keep. That raised a distant converted pawn from 8.7 to 29.8 ms (debug build), and the rule
above costs 11.0 ms there.

**Footprint and tiles.** A splat is cut at the squared Mahalanobis radius

$$r^2_{\max}(\alpha) = \min\big(2\ln(255\,\alpha),\ 9\big),$$

which is where its own contribution falls below 1/255, and never beyond 3σ. The tile count is exact rather than
taken from a bounding box. For each tile in the axis-aligned extent, the minimum of the quadratic form
$q(\delta)=\delta^\top \Sigma'^{-1}\delta$ over the box of the tile's pixel centres is computed in closed form: since
$q$ is convex, it is either 0 or lies on an edge, where it is a clamped 1D quadratic. The emit stage repeats the same
test. Because the blend cuts at the same radius, no pixel outside a counted tile could have been touched, so tile
edges are seamless. A culled slot writes the reason for culling into its unused depth-key word (edit, depth, no area,
faint, off-screen, no tile), and the viewer's Gaussians panel counts these reasons without adding a counter to the
projection.

**Depth key.** The key is `bit_cast<uint>(z) >> 7`, which is order-preserving for positive floats and 24 bits wide.

### 2.5.2 Blending

One 16 × 16 thread group per tile loads the tile's records in batches of 256 into group memory, and every pixel walks
them from there. For pixel $p$ and splat $i$ with conic $Q_i=\Sigma_i'^{-1}$, centre $\mu_i$ and colour $c_i$:

$$\alpha_i(p) = \min\!\Big(0.99,\ \alpha_i' \exp\!\big(-\tfrac12 (p-\mu_i)^\top Q_i (p-\mu_i)\big)\Big),\qquad
C = \sum_i c_i\,\alpha_i \prod_{j<i}(1-\alpha_j) + T\,C_{\text{behind}}.$$

Contributions below 1/255 are skipped. A pixel stops once $T=\prod(1-\alpha_j) < 10^{-4}$, and the group stops loading
once all 256 of its pixels are done. $C_{\text{behind}}$ is either the background or the opaque *under* layer of
meshes and points, composited by view z: the walk ends at the first record behind the layer's depth. Depth is written
either as the alpha-weighted mean or as the first crossing of a coverage threshold. The same walk also accumulates a
six-rank Cryptomatte, by id coverage, and for pruning the per-splat contribution $T\alpha$ or the squared error the
splat's removal would leave (the GaussianPOP score). Specialised entry points compile the composite and matte
branches out, because leaving them live in a frame that has neither cost the blend half again.

**Surfels and harder kernels** (athenea-pieces branch). A Gaussian with one scale exactly 0 is a *surfel*. The
rasteriser draws it by intersecting each pixel's ray with the disc's plane (Huang et al. 2024): the disc's axes are
stored in homogeneous screen coordinates and pulled back through the pixel's two planes. The screen filter is a max,
not a convolution: the larger of the disc and a round Gaussian of variance ½ px². Its opacity is paid back by area,

$$\alpha' = \alpha\cdot\min\!\Big(1,\ \frac{\sqrt{\lambda_1\lambda_2}}{\sqrt{\lambda_1\lambda_2}+\tfrac12-\sqrt{\min(\lambda_1,\tfrac12)\min(\lambda_2,\tfrac12)}}\Big),$$

where $\lambda_{1,2}$ are the eigenvalues of the undilated screen covariance. A merged Gaussian may carry an opacity
$D>1$. That value selects Spark's harder kernel, $1-(1-g)^A$ with $A=\exp((D^2-1)/e)$ and $D$ capped at 5. This is the
composite of $A$ coincident Gaussians, and its cut is widened by $0.7(D-1)\sigma$. A *hard surfel* uses
$\exp(-(|u|^p+|v|^p)/2)$, with $p$ fitted to its members' kurtosis. Section 6 covers both, together with the merges
that produce them.

### 2.5.3 Measured

**Coverage of converted surfaces** (`athenea_coverage_tests`, 256 × 256, alpha of planes converted by mesh2splat,
against the mesh's alpha; sizes are fractions of full screen):

| Plane | Mesh | Raster 1× / 0.25× / 0.05× | Traced 1× / 0.25× / 0.05× |
|---|---|---|---|
| constant 0.25 | 0.250 | 0.251 / 0.250 / 0.248 | 0.251 / 0.251 / 0.251 |
| constant 0.5 | 0.500 | 0.501 / 0.499 / 0.493 | 0.501 / 0.501 / 0.500 |
| constant 0.75 | 0.750 | 0.752 / 0.749 / 0.736 | 0.752 / 0.752 / 0.752 |
| opaque | 1.000 | 1.000 / 1.000 / 0.999 | 1.000 / 1.000 / 1.000 |

Without `paidOpacity`, the raster read 0.18 / 0.455 / 0.73 at 0.05×.

**Motion blur and lens** (`[motion]`, `[lens]`; truth = the same rasteriser at N instants, or 64 lens positions,
averaged on the device):

| Case | Result |
|---|---|
| shutter step 0.10 (many splat widths) | p99 8, max 24 |
| shutter step 0.03 | p99 2, max 9; energy ratio 1.0000 |
| shutter step 0.9 | energy ratio 0.9030 (tails below the 1/255 cut) |
| zero shutter vs still frame | max 0, bit for bit |
| f/2, 12.7 px confusion | p99 17 (not blurring: p99 113) |
| f/11, 2.3 px confusion | p99 9 (not blurring: p99 33) |
| at the focus distance | max 0, bit for bit |

**Frame cost by stage** (`ATHENEA_STAGES=1`, every stage waited for; the sparrow's clips, 5 887 323 Gaussians at
1600 × 1000, playing):

| Stage | Before | After | Change |
|---|---|---|---|
| depth sort (24 bits) | 11.3 ms | 6.6 ms | tiled radix |
| tile sort (8.8 M pairs) | 8.4 ms | 6.0 ms | tiled radix |
| blend | 13–25 ms | 8–13 ms | tile records shared through group memory |
| project, emit, counts, commit | 13.6 ms | 13.4 ms | – |
| frame, playing | 102 ms | 49 ms | cloud no longer re-uploaded each step |

A frame of six million Gaussians spends its time on the Gaussians rather than on pixels: changing the render scale
from 800 × 500 to 3200 × 2000 leaves the time unchanged.

## 2.6 The Gaussian ray tracer

`render::GaussianRayTracer` evaluates each particle in 3D at its peak response along the ray, with no EWA
approximation and no 0.3 px dilation. Particles are blended front to back in order of their peak along the ray. In
the particle's whitened frame (origin $o$, direction $d$, with $\Sigma^{-1/2}$ applied), the peak and the entry into
the cut ellipsoid are

$$t^\ast = -\frac{o\cdot d}{d\cdot d},\qquad \rho = \lVert o + t^\ast d\rVert^2,\qquad
t_{\text{entry}} = t^\ast - \sqrt{(r^2_{\max}-\rho)/(d\cdot d)}.$$

The textbook discriminant $b^2-ac$ cancels catastrophically for flat particles and lost 176 of 400 thin layers, which
is why the peak form is used. Colour is the rasteriser's: the harmonics evaluated once per frame in the direction
from the eye to the centre.

**Two routes, one integrator.** Both routes feed the same `Integrator` (`rt_integrate.slang`). The *Hardware* route
gives every particle a stretched-icosahedron proxy (3DGRT's) in a device BVH queried with inline `RayQuery`. The
*ComputeBvh* route is a Karras LBVH built on the GPU (Morton codes, radix sort, hierarchy, refit until a pass changes
nothing) and traversed in compute. Each traversal records up to 256 unsorted entries, which are evaluated and ordered
after traversal by a 128-bucket counting sort and then an insertion sort over an index permutation. When a ray meets
more proxies than that, the segment closes at the nearest dropped entry and the next traversal resumes a relative
10⁻⁵ before it. Particles peaking beyond the boundary are carried into the next segment; only when more than 64 are
carried are the nearest blended early, which is the route's single approximation.

| train_7k (742 k splats), 1080p | Frame | Build |
|---|---|---|
| Hardware (Metal) | 435 ms | 731 ms |
| ComputeBvh (Metal) | 343 ms | 168 ms |
| tile rasteriser, same frame | 13 ms | – |

On Metal, each non-opaque candidate costs a round trip out of hardware traversal. An opaque closest-hit query over
the same 15 M proxy triangles takes 6 ms, but enumerating every candidate takes 255 ms, so `Auto` picks ComputeBvh on
Metal. The optimisation history on this frame went from 2.52 s (a 3DGRT-style 16-entry k-buffer) to 343 ms. The
largest steps were recording unsorted entries and evaluating after traversal (1.94 s → 658 ms) and the bucket sort
(−85 ms); a record size of 256 was chosen from a sweep of 64 to 512.

**Shadow and secondary queries.** Transmittance is a product, so `rt_shadow.slang` takes proxies in whatever order
they arrive, with no k-buffer, and stops when the product falls below its cut. It does not cull back faces, because a
shadow ray born inside a proxy must still see the particle it starts in. Three closed forms check it: $1-\alpha$
through one centre, measured 0.50000; $(1-\alpha)^4$ through four, measured 0.06250; and early termination after
three of four. The integrator answers a query `rtTraceSegment(ray, [a, b))` rather than writing a pixel, and
`segmentOver` composes two segments: a sparse cloud split at any depth is bit-exact with the single query.

## 2.7 Points

`PointRasterizer` draws point clouds through a z-buffered raster pipeline with no sort, optionally followed by
surface splatting and eye-dome lighting. On devices without rasterisation (CUDA), points are drawn as discs in the
splat pipeline. In that case a record's `colour.w > 0.5` marks it as an opaque disc, which ends the walk exactly as
the under layer does.

## 2.8 Meshes: geometry, scene and visibility

**On the device, in Hydra's order.** `geom::MeshBuilder` triangulates Hydra topology in `HdMeshUtil`'s fan order,
including holes and left-handed orientation. It computes smooth normals with `Hd_SmoothNormals`' formula, as
per-corner cross products scattered to points through a radix sort, and expands primvars of every interpolation per
corner, resolving indexed primvars and decoding doubles on the device. A height field's normals come within
1.1·10⁻⁷ rad of a brute-force sum. `world::GpuScene` pools positions, indices, corners, faces and primvar values,
with a 64-byte record per mesh and a 176-byte `InstanceRecord` per drawn copy. `world::Instancing` composes instancer
chains exactly as Storm does, $\text{instancer}\cdot T\cdot R\cdot S\cdot\text{instanceTransform}$, nested as
$\text{parent}[i]\cdot\text{level}[j]$, with one dispatch for all instanced sets. For Kitchen_set_instanced's 1462
sets this took command recording from 32 ms to 0.4 ms.

**Three routes, one visibility buffer.** Each route writes (instance + 1, triangle) per pixel. Shading and AOVs
rebuild the hit from those two numbers with Möller–Trumbore in view space (`technique/surface.slang`), so all routes
shade alike.

- `VisibilityRaster` uses reversed infinite Z (depth $= \text{near}/z$, D32Float), one draw per mesh with
  `instanceCount`, and a single shared root object.
- `VisibilityTrace` builds one BLAS per mesh and a TLAS per frame. A kernel writes the instance descriptors in each
  backend's layout (64 B generic, 80 B OptiX, 68 B Metal).
- `VisibilityBvh` builds a Karras LBVH per mesh and one over instances, for devices without RT hardware.

Single-sided culling matches Storm in all three routes.

**Storm as the oracle** (Kitchen_set, 480 × 270; ids compared as segmentation, because Storm numbers prims
differently):

| Route | Coverage | Segmentation | Depth, worst | Neye > 6/255 |
|---|---|---|---|---|
| raster | 0 differ | 18 of 57 219 | 1.2·10⁻⁶ | 17 |
| rays | 4 differ | 41 of 57 236 | 6.3·10⁻⁶ | 17 |
| bvh | 4 differ | 36 of 57 233 | 6.3·10⁻⁶ | 17 |

**Visibility timings** (`athenea stage --frames 40`, median frame after the first, headlight shading, readback
included):

| Scene | Size | raster | rays | bvh |
|---|---|---|---|---|
| Kitchen_set (1788 meshes) | 480 × 270 | 18.0 ms | 4.8 ms | 8.9 ms |
| Kitchen_set | 1920 × 1080 | 31.7 ms | 19.2 ms | 29.2 ms |
| Kitchen_set_instanced (1462 sets) | 1920 × 1080 | 29.1 ms | 19.2 ms | 28.6 ms |

Raster loses mainly on the host side: recording about 1800 draws costs about 8 ms, because slang-rhi writes render
state per draw. The automatic setting therefore prefers rays wherever the device has ray queries.

## 2.9 Materials

**Textures.** The `TextureStore` uses Hio only to read a file's bytes. The bytes are uploaded raw and a kernel decodes
them into the working space. Mips are a kernel that decodes sRGB, filters in light and re-encodes. UDIM is an
indirection table. All textures sit in one `ParameterBlock` of 1024 slots, which Metal takes as an argument buffer.
Filtering is chosen per target with `__target_switch`: `SampleGrad` where compute allows it, and an explicit level
from the footprint's wider side elsewhere (CUDA). On Metal, where both are available, they pick the same levels. A
decoded texture is the file to the last bit (0 of 3404 components differ). Input colour spaces resolve through
`colour::ColourNames`, and transforms are generated by OpenColorIO (§2.12).

**MaterialX into Slang.** The generator derives from MaterialX 1.39.5's `SlangShaderGenerator`. Every node keeps its
genglsl or genslang implementation except four kinds whose rasteriser meaning cannot hold here:

- the surface node, which has no light loop here: its BSDF graph runs once, pushing lobes;
- BSDF and EDF nodes, which push lobes instead of answering a light;
- image nodes, which sample the texture table;
- `heighttonormal`, which needs screen derivatives.

A BSDF value is a *weight per built lobe*, so `mix`, `layer`, `add` and `multiply` combine weights. Uniforms are not
baked into the code: each input is read from a float blob, and the module is named by a hash of its source, so
materials that differ only in values share one compiled module. Generated sizes include UsdPreviewSurface at 580
lines and 23 blob words, standard_surface at 821 and 59, and OpenPBR at 945 and 55.
`technique::MaterialPrograms` generates one module that imports every compiled material and dispatches on a material
row. A kernel that evaluates materials walks its pixels in quad order (`atheneaQuadPixel`), because on Metal
the lanes run along rows, and in row order every vertical derivative is wrong (15 939 of 15 939 threads).

**The lobe library.** It contains Oren–Nayar and energy-compensated EON, Burley, translucent, dielectric (reflection,
transmission, both), conductor, generalised Schlick with an F82 tint, and sheen in both its Imageworks and Zeltner
forms. Each lobe has `eval`, `sample` and `pdf`. Microfacet lobes sample the visible normal distribution and
transmission follows Walter. A material's lobes form a `LobeStack`, which is sampled with one-sample MIS by the lobes'
weighted directional albedos.

**Checks against MaterialX itself.** The same graphs are compiled a second time in a reference variant whose closures
are MaterialX's own genglsl responses, and a kernel evaluates both for 65 536 light directions. The worst component
difference over eleven graphs is **3.2·10⁻⁵**. On the device, the lobes pass a χ² test of sampled directions against
the pdf (393.2 on 399 degrees of freedom for the diffuse lobes) and a white furnace that agrees within 3 %.

**Cutouts and bump.** Visibility evaluates `opacityThreshold`: the raster uses a generated discarding fragment
shader, and the ray routes carry the ray past a cut sample up to sixteen times. Only rows flagged as cutouts pay for
this. A cutout square shows what lies behind it exactly (8464 pixels, 0 mismatches) on all three routes.

## 2.10 Lights

**Records and units.** Each UsdLux light becomes one record in world space, in the units the stage authored.
Exposure, the blackbody of a colour temperature (Krystek's fit, normalised to luminance 1) and `normalize` are
derived in the shader. Supported kinds are sphere, disk, rect, distant with an angle, dome and cylinder, plus
shaping cones and IES profiles. An IES profile is read as authored, sampled bilinearly with its declared symmetry,
and modulates radiance but never density. Lights under instancers are composed on the device as meshes are.

**Sampling where the light stands.** A sphere or sun is sampled uniformly over the cone it subtends. A disk, rect or
cylinder is sampled uniformly by area and converted to solid angle by $p_\omega = d^2/(\cos\theta_l\,A)$. A dome is
sampled either cosine-weighted about the shading normal or by warping down its image's mip chain by luminance. The
warp's density needs no tables:

$$p(\omega) = \frac{L(u,v)}{\bar L\; 2\pi^2 \sin\theta},$$

where $\bar L$ is the 1 × 1 mip level's mean. The image's variation chooses between the two strategies: on a flat sky
the warp is 20 % out where the cosine is 0.08 %. `lightPdf` returns the density of any direction and `lightHit`
answers the converse question, which light a ray meets and with what radiance. Together they are what MIS needs.
With `athenea:chooseLights`, a sample takes one light chosen through a **light BVH** (`light_bvh.slang`). Each node is
a box, an orientation cone (Conty and Kulla 2018) and the power beneath it, and the BVH is built by the same compute
LBVH kernels as the geometry.

**Checks.** Every light is held to a closed form that shares no code with the renderer: a Lambert plane under a
sphere, a disk, a rect (Lambert's edge formula), a sun and a dome. In each case 0 of 8281 pixels are beyond 2 %;
the worst is 1.6 % for the disk. Each light also runs a χ² test with 10⁶ samples, binned in the light's own support,
against `lightPdf` integrated per bin: sphere z = −0.23, disk 0.14, rect −0.27, sun −0.29, dome −1.61, every pdf
integrating to 1.0000. The dome warp needed two fixes, both found by χ² in image space. The bug reached z = 61 543
and later 1 164 589, against 0.84 after the fixes. The per-sample consistency check had passed throughout, because it
compares a density with itself.

**Measured** (Kitchen_set with four authored lights, 1600 × 900, `athenea view --frames 200`, draw median):

| Configuration | Draw |
|---|---|
| headlight only, rays | 10.61 ms |
| four lights, 1 sample, rays | 27.77 ms |
| the same, compute BVH | 35.14 ms |
| the same, raster | 40.94 ms |
| four lights, 16 samples, rays | 275.40 ms |

## 2.11 The path tracer

The path tracer is the engine's ground truth. Its paths start from the same visibility buffer the raster shading
reads, so the two techniques differ by exactly one thing: the bounces. With a lens, the tracer additionally casts its
own primary rays. A hit is rebuilt by `material_surface.slang`, its material is evaluated into the same lobe stack,
and direct light is gathered by next-event estimation (NEE). The bounce samples the stack, and the path throughput is
multiplied by `LobeSample.weight` $= f|\cos\theta|/p$ (Figure 4).

![Figure 4. One vertex of the path tracer: the hit is rebuilt from the visibility buffer, direct light is gathered by next-event estimation, the lobe stack is sampled for the next bounce, and both estimates are combined by multiple importance sampling.](figures/fig-pathtracer.svg)

**MIS.** At each vertex that continues the path by sampling its material, NEE's sample of light $k$ and the material
sample's hits on lights (`lightHit`, including escaping rays meeting domes and suns) are weighed by the power
heuristic,

$$w_{\text{light}} = \frac{p_l^2}{p_l^2 + p_m^2},\qquad p_l = P(k\mid x)\; p_k(\omega),$$

where $P(k\mid x)$ is the light-choice probability (by power, or through the light BVH) and $p_m$ is `stackPdf`.
Delta lights, delta lobes and the last vertex keep a weight of 1. **Emissive triangles** form a light table
(`technique::EmissiveTable`): each material row's emission is probed once, triangles are weighted by world area times
that luminance, and the choice between lights and triangles is made by power, with each side's density carrying the
other's share. On a floor of roughness 0.2 at 32 paths, MIS reduced error against a deep frame by 4.5× under a rect,
57.7× under an imageless dome and 2.4× under a sphere. An emissive quad and an equivalent UsdLux rect agree to
relMSE 4.1·10⁻⁷ at 8192 paths.

**Accumulation, sampler, adaptivity.** A frame is a running mean of colour × opacity. An earlier implementation took
the product of means, which is unbiased only while every sample is opaque. The sampler folds pixel, stream, absolute
path index, bounce and dimension through a PCG hash, so 256 paths in one pass and 4 × 64 differ by relMSE
3.9·10⁻¹⁵. Between passes the camera is jittered by a Halton (2, 3) offset inside the pixel; the first pass takes no
offset. This is antialiasing at no cost. Adaptive sampling stops a pixel once

$$\frac{\sqrt{(\mathbb E[l^2]-\mathbb E[l]^2)/N}}{\mathbb E[l]} < \varepsilon$$

after a minimum count. The variance used is the larger of the pixel's own and the mean of its 3 × 3 neighbours',
which reduced the stopped pixels beyond three of their own σ from 526 to 100 of 4212. **Thin lens and radial
distortion** come from `UsdGeomCamera`: the lens radius is $f/(2N)$, and rays are scaled by $1+k_1r^2+k_2r^4$.

**Verification.** Every check is a GPU kernel against a closed form or against a second, independent estimator:

| Check | Result |
|---|---|
| closed emissive sphere, $E(1+\rho+\dots+\rho^N)$, N = 0, 1, 2, 3, 6 | 0 of 3072 pixels beyond 10⁻⁴, worst 1.7·10⁻⁷ |
| white furnace under an imageless dome vs unweighted NEE | p99 relative 0.0000, max 0.0003 |
| one bounce vs raster direct light (nothing to bounce off) | p99 1, max 1 (8-bit codes), 4096 paths |
| error vs N, pairs of independent estimates, 64–1024 paths | exponent −0.499 (window −0.42 to −0.58) |
| in focus vs pinhole (lens radius 0.2) | relMSE 0, bit for bit |
| out of focus vs circular-segment profile | 0 of 4235 pixels beyond 4 % |
| OIDN, 16 paths vs 4096-path reference | relMSE 3.65·10⁻⁴ → 7.17·10⁻⁵ |

The furnace exposed a one-sided MIS weight. On a dome against a Lambert lobe the two densities are the same function,
so the weight was exactly ½, and the furnace read 0.5453 instead of 1. The shell exposed a double division,
$\rho\pi/\cos$ instead of $\rho$, whose infinite second moment had held the convergence exponent at −0.43 to −0.45.

**Denoiser.** `technique::Denoiser` runs OIDN 2.5.1's `RT` filter on the engine's own queue, guided by the first hit's
albedo and normal. On Metal, OIDN shares only hazard-tracked buffers, which slang-rhi forbids. The engine therefore
copies through tracked staging buffers using a kernel, since slang-rhi's blit silently did nothing on a wrapped
buffer (4095 of 4096 words untouched). On CUDA the engine's buffers are shared directly. On Vulkan, OIDN runs through
CUDA on imported memory (`oidnNewSharedBufferFromFD`).

**Measured** (Kitchen_set, 1920 × 1080, `athenea stage --frames 11`, median, release):

| Stage | raster | rt (1 path, 1 bounce) |
|---|---|---|
| no lights (headlight) | 43.1 ms | 333.3 ms |
| lit (dome, rect, two spheres) | 96.0 ms | 358.9 ms |

Through MCP, the OpenChessSet path traced at 640 × 360 with 64 paths, one bounce and ACES 2.0 takes 25.7 s.

## 2.12 Display and colour

`technique::DisplayTransform` writes the window's surface texture directly. It supports Standard; AgX, in Wrensch's
analytic fit (log2 over [−12.47, 4.03] stops, a sixth-order sigmoid); and ACES 2.0. The ACES 2.0 output transform
(`aces2.slang`) is ported function for function from the Academy's CTL, including Hellwig 2022's appearance model,
tonescale and gamut compression. The reference's init-time tables (362 entries per hue) are built on the device by
three kernels. Checks: 256 greys over sixteen stops stay neutral to 5·10⁻⁶; 18 % grey reads 0.10000 of reference
white at 100 nits; inverse-then-forward over 8192 colours returns within 6·10⁻⁶ of peak. The display formulas,
re-derived in a second kernel, agree to 3.5·10⁻⁶. OpenColorIO is used **as a compiler**: `colour::ColourCompiler` asks
OCIO for HLSL, loads it as a Slang module `athenea_cs_<hash>`, and fills its LUTs on the device. The host computes
shader text and LUT values but nothing per pixel.

## 2.13 USD and Hydra

The `hdAthenea` plugin implements `HdRenderDelegate` with rprims for meshes, basis curves, points, `ParticleField`
(Gaussian splats) and volumes; sprims for cameras, materials, the six UsdLux light types, ext computations and
coordinate systems; and bprims for render buffers and render settings. Implicit surfaces (sphere, cube, cone,
cylinder, capsule, plane) reach the delegate as meshes, converted by hdsi's scene indices. **Threading:** `Sync`,
which Hydra may run on any thread, only hands CPU records to `Engine` under a lock. The render pass thread commits
uploads, opens `.athc` files and compiles materials, then renders, so the device has a single caller. A skinned cloud
whose arrays carry no time samples is held across steps (`_held`) and not re-uploaded; this took the sparrow's
playing frame from 102.4 to 71.7 ms before the sort and blend work of §2.5.3. **Outputs:** colour, depth, primId,
instanceId, elementId, Neye, normal, `primvars:*`, Cryptomatte and light groups. They are converted on the device
(`usd/aov_convert.slang`) only when a host maps a buffer, and are stored bottom row first, as Storm stores them.
`StageRenderer` is the programmatic entry point and draws until a path-traced frame holds its total.

The engine reads codeless schemas of its own (`AtheneaSplatLightingAPI`, `AtheneaStreamedAssetAPI` and others).
Render settings arrive as `athenea:*` settings: technique, visibility route, path samples, bounces, totals, MIS,
denoise, antialias and lens.

## 2.14 Hosts and front ends

- **`athenea view`** keeps frames on the device: `StageRenderer::draw`, then `DisplayTransform` writes the surface
  texture and Dear ImGui draws on the same device. Measured with materials at 1600 × 900, Kitchen_set draws in
  9.27 ms on rays. The panels are described once in `modules/ui`, so the iOS app draws the same ones.
- **AOFX host.** The AOFX SDK and host are aopenfx's (`third_party/aopenfx`), and the engine declares only what it
  provides (`Host.cpp`). They change only additively under the ABI's rule, so the compositor's bundles load
  unchanged. `aofx_sdk_manifest` fails on any header change. Data-processing steps such as mesh2splat (Section 3) and
  `athenea compare` are AOFX bundles.
- **MCP server** (`athenea-mcp`): JSON-RPC 2.0 over stdio, with a warm device and stage. Tools are `open_stage`,
  `render` (returning a PNG image block), `pick`, `bounds`, `convert`, `timings` and `settings`. A protocol test suite
  checks handshake, notifications and error codes. For a program that composites the frame, `render` takes a camera as
  a compositor holds one (camera to world and a lens in millimetres) and writes the frame's own numbers to a file:
  premultiplied RGBA, view depth, and on request the world position (made on the device from the depth) and the three
  Cryptomatte layers, with the matte's manifest in the answer. `sky`, `skyRotation` and `skyVisible` put another image
  on every dome of the stage, turn it, and hide the background while it still lights.
- **OpenFX node** (`Athenea.ofx.bundle`, built on Natron's OpenFX). A generator that draws a cloud through its camera
  at the project's format and the frame's time. It loads none of the engine: it starts `athenea-mcp` as a child, one
  for the host process, and reads each frame back from a file, so the host's own USD and TBB never meet the engine's
  and a device fault ends the server, not the host. Its knobs are a Camera node's (translate, rotate in an order,
  focal, apertures), the sky, its rotation and visibility, and the layer a host without planes shows in RGBA. Through a
  minimal OpenFX host its frame is bit-identical to the server's own render of the same camera (relMSE 0). In Natron,
  which takes custom planes, a pawn at 1920 × 1080 writes one 32-bit EXR of 21 channels (colour, `depth.Z`, `P`,
  `CryptoObject00`–`02`, Figure 5) whose colour and position match that host's to within $10^{-6}$. The Corvette's relightable
  cloud (3 M Gaussians, 2.9 GB) at 1920 × 1080 with every plane takes 7.5 s for the first frame, which opens and
  uploads the stage, then 120–170 ms a frame under a still sky and 190–220 ms with the sky turning every frame. Nuke
  takes from an OpenFX node only its own planes, so there one layer a node is shown.
- **Blender add-on** (`athenea_hydra`). A Hydra render engine for Blender 5.3 over the engine's delegate: viewport and
  final render by athenea, mesh-to-Gaussian conversion of selected objects, clouds referenced as Empties, the World's
  environment as the dome.

![Figure 5. The layers the OpenFX node gives Natron for one frame of the pawn, written as one 32-bit EXR of 21 channels (1920 × 1080, kloofendal turned 120 degrees): the premultiplied colour, the view depth `depth.Z`, the world position `P` (shown as $12x + 1/2$, $6y$, $12z + 1/2$) and the three Cryptomatte layers, shown here by the engine's preview of the same matte, one colour per id (the body and the head).](figures/img-ofx-layers.jpg)

## 2.15 Verification without a CPU oracle

Every image comparison is a kernel. `ReferenceRenderer` is the ground truth for splats: every pixel considers every
splat, sorts exactly what reaches it (depth, then index, the rasteriser's tie rule) and blends front to back, with
no tiles, ranges or pairs. A comparison is valid only when no pixel overflows its 1024 contributors.
`renderPeaks` is the ray tracer's reference, sorting exactly by peak. The metrics read back to the host are
`compareImages` (p99, max and the number of pixels over 2, in 8-bit sRGB codes), `compareHdr` (relMSE
$=\overline{(a-b)^2/(b^2+10^{-2})}$ and the relative p99) and `countDifferent` for id buffers. The ray tracer is held
to its reference at p99 ≤ 1 with at most 0.05 % of pixels over 2. Against the rasteriser, the tolerance holds only
where the two definitions agree: p99 2 for orthographic sparse scenes, where EWA is exact, and 4 for perspective
scenes with particles a few pixels across. Dense scenes differ by tens of codes, which reflects ordering rather than
error.

**Three devices** (each machine computes the same GPU metric itself, and no pixels travel between
machines):

| | Metal (M5 Pro) | Vulkan (L4) | CUDA (L4, OptiX 9, no inline rays) |
|---|---|---|---|
| Cases | 188 | 188 | 188 |
| Passed | 188 | 180 | 96 |
| Failed | 0 | 0 | 1 |
| Skipped (capability named) | 0 | 8 | 91 |

## 2.16 Limitations

- Splats are not yet inside the path tracer's secondary rays. A splats-only stage is traced whole by
  `GaussianRayTracer`. With meshes, splats are composited over the path-traced surfaces by depth; they cast shadows
  onto meshes, but a bounce ray does not see them.
- The rasteriser's motion and lens blur are moment-matched Gaussians, not plateaus or bokeh discs: up to 38 % too
  peaked for long streaks.
- Clouds whose Gaussians are a few hundredths of a pixel across vanish from the raster (5·10⁻⁶ of the frame against
  3·10⁻⁴ for the mesh and the tracer); levels of detail are the remedy (Section 6).
- Raster mesh visibility pays about 8 ms of host time per 1800 draws.
- Material compilation is synchronous.
- The TLAS is rebuilt every frame.
- Lens distortion beyond $k_1, k_2$ is not applied.
