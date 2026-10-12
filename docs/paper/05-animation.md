<!-- Copyright (c) 2026 jesus luque. -->

# 5. Animated assets: skeletons, skinning and per-part visibility (the pawn and the sparrow)

This section describes how athenea turns a rigged USD asset (UsdSkel) into a
cloud of 3D Gaussians that a skeleton still drives. It covers how the cloud is
converted, how it is posed on the GPU every frame, how its light and
self-shadowing are kept valid under deformation, how levels of detail treat it,
and how it is played back in a browser and on a phone. Two assets run through
the section. The Eurasian tree sparrow (`SparrowBird.usda`, 609 joints, 71
animation clips) is the animated test case. The chess pawn of The Open Chess
Set is the rigid control. Every number below is taken from
`docs/decisions.md`, the source files named alongside it, or the measurement
files of the web player (`sparkwebGPU`). Numbers that were not measured are
marked as not measured.

## 5.1 Design principle: carry the rig, not the frames

There are two ways to animate a Gaussian cloud. The first stores every
gaussian's position, orientation and scale at every frame. The second stores
the gaussians once, in the bind pose, together with the joints that carry
each one, plus one transform per joint per frame. athenea uses the second, for
three reasons, each measured:

1. **Size.** For a plausible character of 150 k gaussians over 120 frames,
   the per-frame arrays cost 960 MB at half precision (1.2 GB at float). The
   rig (four joints per gaussian, 60 joints per frame) costs 2.9 MB. A
   synthetic tube (200 rings × 64 segments, 40 joints, 184 320 gaussians) was
   measured as written. Its file is 9.27 MB at 25 instants and 9.50 MB at 121,
   against about 713 MB for the same range as per-frame arrays. Five times the
   frames cost 230 kB.
2. **Exactness between samples.** A rig is exact at every instant of the
   range. A sampled cloud is only as good as the reader's interpolation.
3. **Stable identity.** The posed cloud is written into buffers that outlive
   it, so a renderer sees the same cloud from one frame to the next. That is
   what lets the ray tracer *refit* its acceleration structure instead of
   rebuilding it (§5.6).

The cost of this choice is that whatever was baked in world space (radiance,
spherical-harmonic transfer, visibility) stops being valid as soon as a limb
moves. §5.7 and §5.8 deal with this.

## 5.2 Reading a rigged stage

The converter (`athenea mesh2splat`) does not go through Hydra. It reads
meshes through its own `MeshStage`, which leaves two ways to handle a skinned
mesh:

- **Posed.** Without `--skinned`, the stage is posed before it is read.
  `UsdSkelBakeSkinning` writes the posed points as time samples into the
  *session layer*, over `GfInterval(t, t)`. This costs one pose, leaves the
  file on disk unchanged, and leaves a stage with no `SkelRoot` untouched.
  Test: a square carried by one joint that slides (1.2, 0.4, 0) between t = 0
  and t = 1 converts to bounds, folded on the device, that differ by exactly
  that slide.
- **Carried.** With `--skinned`, the **bind** pose is read and
  `UsdSkelBakeSkinning` is skipped. Posing first would skin the geometry
  twice. For each mesh, `MeshStage` resolves its skel binding with a
  `UsdSkelCache` over every `SkelRoot` and one `UsdSkelSkinningQuery` per
  skinnable prim. It returns the joint influences, `geomBindTransform`, the
  skeleton path and the skeleton's joint order.

**Joint order.** A mesh may list its own subset of joints in `skel:joints`.
UsdSkel's mapper only maps skeleton order to mesh order, so it is run over the
identity to learn the skeleton index of each mesh index. Influences are always
written in skeleton order, so one cloud has one joint order no matter how many
meshes it was converted from. A mesh in a skinned stage that no joint carries
(the sparrow stage holds a cylinder and a plane beside the bird) still gets
one influence record per gaussian: four joints of zero weight, which the
skinner reads as "stay in the bind pose". This keeps rig and cloud aligned
one to one.

**`--skinned` forces `--no-bake`.** A radiance bake holds the scene's
environment and bounces (the ground under a foot), and carrying that up with
a lifted limb amounts to rotating a lightmap. A skinned cloud carries its
material and is relit every frame. Its light transfer, when one is asked
for, is kept in a form that rotates with the gaussian (§5.7).

## 5.3 Influences per gaussian, and their gradients

`mesh_pack.slang` writes each triangle corner's four heaviest influences into
a second picture laid out exactly like the mesh's, so the AOFX effect
addresses both the same way. The effect knows where inside the triangle each
gaussian stands, so it blends the corners there
(`blendInfluences`, `mesh2splat.slang`). With barycentric coordinates
$\lambda_a,\lambda_b,\lambda_c$ and corner weights $w_{cj}$, the raw weight of
joint $j$ is

$$\tilde w_j = \sum_{c\in\{a,b,c\}} w_{cj}\,\lambda_c .$$

Twelve (joint, weight) pairs go in and four come out. A joint already present
gains the weight, an empty slot takes a new joint, and otherwise the new joint
replaces the lightest. Once the four joints are chosen, each kept weight is
recomputed as its full blend and the four are renormalised,
$w_k = \tilde w_k / W$ with $W=\sum_{i\in K}\tilde w_i$. A gaussian is
therefore always fully carried.

**Gradients.** $\tilde w_j$ is linear on the triangle, with
$\nabla\tilde w_j=\sum_c w_{cj}\nabla\lambda_c$, where
$\nabla\lambda_b = (e_2\times n)/|n|^2$,
$\nabla\lambda_c = (n\times e_1)/|n|^2$ and
$\nabla\lambda_a = -\nabla\lambda_b-\nabla\lambda_c$. The gradient of the
renormalised weight follows from the quotient rule:

$$\nabla w_k = \frac{\nabla\tilde w_k - w_k\sum_{i\in K}\nabla\tilde w_i}{W}.$$

This is exact at the gaussian's centre. Only the two components along the
gaussian's rest axes $(\hat u,\hat v)$ act on a disc, and the gradients sum to
zero, so only the first three joints' gradients are stored. That is three
words of two halves, **12 bytes per gaussian**, in
`primvars:athenea:splat:jointWeightGradients` (`half[]`, `elementSize` 6). The
fourth gradient is minus the sum of the other three. With gradients, the
effect's record for a carried gaussian grows to nine entries.

**Measured.** A square bound entirely to one joint converts to 1056
gaussians, all naming that joint with full weight. Through the effect, 272
gaussians of a quad whose corners all name joint 3 come out on joint 3, with
the other slots empty and the weights summing to one. With one joint
everywhere, the stored gradient is below $10^{-3}$.

**Truncation.** Four influences is a truncation. On the sparrow, 27 % of wing
vertices carry more than four (6686 vertices, `elementSize` 25), and a
gaussian takes the union of three corners' joints, so the share of truncated
gaussians is at least as large. That share has not been measured.

## 5.4 Skinning a Gaussian on the device

`scene::SplatSkinner` dispatches `shaders/athenea/scene/splat_skin.slang`, one
thread per gaussian. Its inputs are the rest cloud, the (joint, weight)
pairs, the joints' skinning transforms at the current instant, and three
matrices: the bind transform $G$, `skelToWorld` and `worldToPrim`. USD
matrices are row-major and act on row vectors, so the host transposes them as
`geom::Skinner` does for meshes.

### 5.4.1 The mean

Let $q = G\,p$ be the rest point in bind space, $X_j q = A_j q + t_j$ joint
$j$'s skinning transform, and $M=\texttt{worldToPrim}\cdot\texttt{skelToWorld}$.
The posed mean is linear blend skinning, with exactly the arithmetic the mesh
skinner uses:

$$\mu' = M \sum_j w_j\,X_j\,(G\,\mu).$$

### 5.4.2 The frame: the whole Jacobian

The posed covariance should be the rest covariance pushed through the
derivative of the position map. When the weights vary in space, that
derivative is

$$J = L(M)\Big[\sum_j w_j A_j\,L(G) \;+\; \sum_j (X_j q)\,\nabla_p w_j^{\mathsf T}\Big],$$

where $L(\cdot)$ is the linear part. The first term is the usual "blend the
joints' linear parts". The second, the *elastic* term, is what stretches a
gaussian across a bend where two joints disagree about where its point goes.
With two joints a relative angle $\theta$ apart and a weight transition of
length $L$, its size relative to the first term is about
$2\tan(\theta/2)\,d/L$, which is of order one at a right angle.

The kernel builds the elastic term in the same loop that forms $X_k q$. It
accumulates $\sum_k (X_k q)\,g_k$ on each rest axis, *relative to the blended
point*. The gradients sum to zero, so this is the same sum, but it is computed
as a difference of nearby points, which stays exact in float far from the
origin. A joint with zero weight but a non-zero gradient still contributes.

The posed covariance in the gaussian's plane is

$$\Sigma' = J\,E\,S^2E^{\mathsf T}J^{\mathsf T},\qquad E=[\hat u\ \hat v],\ S=\mathrm{diag}(s_u,s_v).$$

The kernel takes an orthonormal basis $(e_1,e_2)$ of the plane spanned by
$a=J\hat u\,s_u$ and $b=J\hat v\,s_v$, writes $[a\ b]=[e_1\ e_2]K$ with $K$
upper triangular, and diagonalises $C=KK^{\mathsf T}$ in closed form. The
larger eigenvalue is $m+r$, with $m=\tfrac12(c_{00}+c_{11})$ and
$r=\sqrt{\tfrac14(c_{00}-c_{11})^2+c_{01}^2}$. The smaller is computed as
$(k_{00}k_{11})^2/(m+r)$ rather than $m-r$, which would cancel to nothing for
a thin gaussian in float. The new axes are the eigenvectors, the new sizes
the square roots of the eigenvalues. The first axis is the eigenvector closer
to the first carried axis, and where $c_{01}$ is pure rounding the basis is
kept exactly, so a round gaussian is not spun. The third axis is the plane's
normal, and the third size (a disc's thickness) is left as converted. The
shading normal turns with the cofactor of the whole $J$, because
$(Ja)\times(Jb)=\mathrm{cof}(J)(a\times b)$.

The kernel writes only position, rotation and the two in-plane scales.
Opacity, colour, harmonics, PBR channels and the word holding the third size
are left untouched. A frame therefore costs one kernel over the cloud and
nothing is decoded again.

### 5.4.3 Measured

The strip test in `tests/scene/test_loading.cpp` uses 4096 gaussians across a
linear ramp of weights, bent a quarter turn. It compares each gaussian's
decoded posed covariance with the analytic
$J=(1-w)I+wR+(Rp-p)\nabla w^{\mathsf T}$, and counts relative Frobenius errors
over 3 %. The tolerance comes from the format: a ten-bit smallest-three
quaternion cannot pin an axis closer than about $\sqrt2/1023$.

| case | blend only, Gram–Schmidt (before) | whole Jacobian, eigenframe (after) |
|---|---|---|
| bent, axes 0.1 rad from the ramp | 2048 of 4096 | 0 |
| bent, axes turned 30° | 2048 | 0 |
| rigid turn and slide (control) | 0 | 0 |
| a joint that shears, constant weights | 4096 | 0 |
| bent, no gradients stored | 2048 | 1984 |

An earlier test placed 4096 gaussians on a helix, each with its own frame and
size. At rest every gaussian stays where it was. Under a rigid turn of 1.2 rad
about a slanted axis plus a slide, every gaussian is carried rigidly. Only
four counters are read back.

On the sparrow's `Flight` clip (61 frames, 1500 wing triangles, the posed mesh
as reference), the blend term alone departs from the mesh's own map by a
median of 4–6 %, a 90th percentile of 14–19 % and a 99th of 37–50 %, on the
wing hand and the feather fans. With the elastic term the 90th and 99th
percentiles fall to 5–6.5 % and 11–13 %. In path-traced images (1 125 022
gaussians, 1024×768, frames 5, 31, 36) the two kernels differ by a few 8-bit
levels in about a tenth of the pixels (p99 14–16), and the wings look the
same. The whole-image relMSE against the mesh gets 2–3 % *worse*, but an
ablation shows this is the metric's doing: negated, swapped, halved and
doubled gradients (0.0900, 0.0907, 0.0887, 0.0983 against 0.0904) rank by
the size of the change, not its correctness, because conversion error
dominates. Correctness rests on the strip test, and on a diagnostic over
174 305 wing gaussians: between neighbours with the same four joints, the
stored gradients predict the change of the stored weights with an RMS
residual of 12.5 %, against 158 % with the axes swapped and 100 % with no
gradient.

## 5.5 The file: the rig as USD

A skinned cloud is a `ParticleField3DGaussianSplat`. The rig was first
written under `AtheneaSplatSkinningAPI`, and is now written under UsdSkel's
own names. `mesh2splat --skinned` applies `SkelBindingAPI` and writes
`skel:jointIndices` and `skel:jointWeights` (`elementSize` 4),
`skel:geomBindTransform`, `skel:joints` and `skel:skeleton`. It also writes
`primvars:athenea:splat:skinningXforms`: one matrix per joint, in skeleton
order, **time sampled**. This is the only property of an animated cloud that
changes from frame to frame. The transforms are
`UsdSkelSkeletonQuery::ComputeSkinningTransforms` at each instant of
`--range START:END[:STEP]`.

**Binding to the Skeleton.** usdSkelImaging builds ext computations only for
meshes, curves and points. It does, however, resolve any Skeleton's
`skinningTransforms` at any instant. `HdAtheneaSkelSplatSceneIndex`
(`modules/usd/src/SkelSplat.cpp`) puts those resolved transforms on the cloud
as `athenea:splat:skinningXforms`, remapped by `UsdSkelAnimMapper` when the
cloud names a subset of joints, and forwards a Skeleton's dirty notices to its
clouds. The primvar that the conversion writes is then a **cache**: a stage
that composes the cloud without its Skeleton still moves, and where both are
present the cache wins. Dual-quaternion skinning (`skel:skinningMethod`) is
not implemented for clouds; such a prim is skinned by linear blend, with a
warning printed once.

**Measured.** 512 gaussians bound to a one-joint Skeleton whose animation
puts the joint at t = 1 where the cache has it at t = 2 draw, with the cache
blocked, the cache's t = 2 frame at t = 1 pixel for pixel (p99 0, max 0). On
the sparrow, the 609 joints the engine receives through the Skeleton agree
with the rig layer's; USD's own transforms against the Blender-written cache
differ by at most $5\times10^{-4}$, and the frames differ by a mean of
$8\times10^{-6}$.

**Time scale.** Without `timeCodesPerSecond` the cloud was read at 24 under a
30 fps stage and lagged the mesh (mean error 0.031 at frame 16, 0.049 at 61);
with the rate written, $1.4\times10^{-5}$ and $1.1\times10^{-5}$.

**Clips.** The sparrow arrives as five FBX files with 71 clips. Each clip
changes only `skinningXforms` (609 matrices a frame), so a clip is a layer
over one shared cloud: 377 MB for all 71, against 22 GB for 71 converted
clouds. `SparrowClips.usda` offers them as a USD **variant set**. The cloud
comes in as a payload, because a sublayer's local opinion would beat the
variant, and each clip comes in as a reference inside its variant. The engine
stays generic: it lists variant sets, selects one in the session layer, and
reports `animationRange()`, the span of authored samples (for example 1..25
for `land_hop_1`, 1..33 for `air_fly_A0`). Choosing a clip reads about 3 MB,
not 213 MB.

| asset (`mesh2splat --skinned`) | gaussians | joints | instants | file |
|---|---|---|---|---|
| rigged tube, resolution 512 | 184 320 | 40 | 25 / 121 | 9.27 / 9.50 MB |
| Khronos fox, resolution 512 | 204 517 | 24 | 28 | 13.35 MB |
| sparrow, resolution 1400 (first conversion) | 4 991 908 | 609 | 33 | — |

## 5.6 The engine: posing, bounds, refit and motion blur

`Engine::carryCloud` pairs the file's two arrays into the (joint, weight)
layout, keeps the uploaded cloud as the bind pose, and runs the skinner into a
separate `GpuSplats` that shares the harmonics and PBR buffers, since the
skinner writes neither. Measuring this exposed two errors. First, a posed
cloud's extent is not the bind pose's, so the extent is now folded again on
the device after skinning. Second, `Engine::bounds()` folded the uploaded
cloud instead of the drawn one.

**A time step is not a new cloud.** At first, every time-code change
re-decoded every array and rebuilt the CPU-side influence pairs (137 MB on
the 4.27 M-gaussian sparrow). Arrays are now compared by identity
(`skinningXforms` excluded), pairs are rebuilt only on upload, and arrays
with no time samples are held (`_held`), since the crate file's compressed
integer arrays arrive in new buffers on every read. Viewer, 400 frames at
1280×720:

| | before | after |
|---|---|---|
| cloud uploads over 400 frames | 400 | **1** |
| raster, median / warm | 134.96 / 31.56 ms | **56.87 / 13.40 ms** |
| `rt`, median / warm | 268.66 / 55.31 ms | **178.99** / 55.05 ms |

**Refit, not rebuild.** A pose changes neither the number of particles nor
their order, so `GaussianRayTracer::sync` refits. On the hardware route this
recomputes proxies and updates each chunk's BLAS in place. On the compute BVH
route it refits one tree height at a time, bottom up, with no read-back. A
full rebuild runs every `refitsPerRebuild` = 240 refits. Against a rebuild
every pose, the images agree to p99 0, max 1. A stale tree reads p99 255.
Sparrow at resolution 256 (300 862 gaussians, compute BVH): 140.9 / 143.1 /
135.6 ms per frame before, 84.7 / 83.9 / 83.7 ms after (−40 %).

**Motion blur from the skeleton.** `ParticleField` samples
`skinningXforms` over the shutter. The skinner blends the second pose in the
same loop (points only, about a third of a kernel) and writes each gaussian's
displacement as two words of halves. The rasteriser turns that displacement
into a rank-one covariance term. Hydra brackets rather than clips the
shutter, so the displacement is scaled by $(t_c-t_o)/(t_1-t_0)$; without that
factor the wings blur twice as far. On the 5.9 M sparrow at 1280×800, a
frame takes 90.0 ms with shutter 0 and 100.5 ms with shutter 0.5 (+11 %).

## 5.7 Light that turns with a limb: zonal transfer lobes

A static cloud can carry a precomputed transfer of nine spherical harmonics
in world space. Under a skeleton those harmonics would keep the sky of the
conversion pose, and rotating them costs a $9\times9$ matrix per gaussian.
athenea instead stores one or two **zonal lobes** per gaussian, with each
lobe's axis expressed in the gaussian's own frame. A zonal function rotates by
rotating its axis, a single $3\times3$, so whatever frame the skinner
produces carries the lobes with it, and no gaussian has to be assigned to a
part.

A lobe with axis $a$ and coefficients $z_l$ is, in harmonics,
$g_{lm}=z_l\,k_l\,Y_{lm}(a)$ with $k_l=\sqrt{4\pi/(2l+1)}$. For a fixed axis
the best coefficients are the projection
$z_l = k_l\sum_m f_{lm}Y_{lm}(a)$. By the addition theorem the residual is

$$\|f-g\|^2=\|f\|^2-\sum_{l=0}^{2} z_l^2,$$

so the axis is chosen to maximise $\sum_l z_l^2$. On the device the search
tries the band-1 direction, the normal, and 64 Fibonacci directions on a
hemisphere, then refines by a walk that halves its step. The second lobe is
fitted to the first lobe's residual, and the two are refitted against each
other twice (Sloan's ZH fit). The data is ten floats per gaussian,
`primvars:athenea:splat:transferZonal` (`elementSize` 10): per lobe, the
octahedral $(u,v)$ of the axis in the frame and $z_0,z_1,z_2$. On the device
it is the ordinary `transfer` buffer with `transferCount` = 10. At read time,
`splatTransferFrame` turns each axis into world space using the gaussian's
current rotation and rebuilds the nine harmonics.

**The bake pose.** At first the bake posed the cloud at `--time` using four
influences per gaussian, while Hydra skinned the mesh with all of its own
(ten per vertex on the feathers, 25 on the wings). Where the two disagreed,
the bake's ray missed its surface, and 566 603 of 1 504 160 gaussians came
out transparent. The bake now holds the skeleton in its bind pose
(`StageRenderer::holdBindPose`) and poses the cloud with identity transforms:
1 504 156 of 1 504 160 find their surface. The transfer therefore holds the
bind pose's self-shadowing (wings spread). Occlusion that another limb casts
in another pose is the job of §5.8.

**Checked.** In `[zonal]`, a two-lobe transfer survives the fit and f16
storage within 0.5 %. Turning a gaussian's frame by $R$ and reading along
$Rw$ gives what the unturned gaussian gives along $w$. In a USD test, 512
gaussians carried by a joint that turns 60° match the same cloud under an
Xform of that rotation, with relMSE below $10^{-4}$ on both raster and traced
routes. The fit's error on the sparrow has not been measured. The indirect
half of the transfer is not kept zonal.

## 5.8 Per-part visibility fields: baked once, a product of table reads per frame

A relit cloud also has to shadow itself, and the measurements ruled out the
obvious routes. In the traced route `--splat-shadows` changed nothing, because
a traced splat had no shadow pass. In the raster route it cost 36× the frame
(49.58 → 1810.24 ms) and crushed the bird to black. Sweeping the self-offset
from 1 to 800 own sigmas moved the mean image monotonically from 0 % to 99 % of
the unshadowed value, with no plateau. The literature agrees that baked
visibility does not survive articulation unless it is factored by body part:
Lin et al. use directional MLPs per part, and DNF-Avatar a scalar AO grid per
bone. athenea combines the two ideas: a **directional table per part**.

**Partition.** `partitionJoints` takes the largest subtree not yet a part,
with at least `minJoints` (6) joints, and makes it a part, until there are
`--parts` (12). On the sparrow this gives root, centre, chest, pelvis, head,
and each wing in three segments. Each gaussian belongs to the part of its
dominant joint.

**Bake** (`athenea visibility`, `splat_visibility_bake.slang`). This is done
once, in the bind pose. For part $p$, a grid of $N^3$ probes covers the
cloud's box plus a 5 % margin. Each probe holds an octahedral map of $O^2$
directions. A texel stores the transmittance of a ray leaving the probe
through part $p$'s gaussians alone. The part's gaussians are gathered into a
cloud of their own and traced with inline rays, using the same product of
$(1-\alpha)$ that the shadow kernels use. Each probe also stores an ambient
term, its mean over directions, used for domes.

**Read** (`splat_visibility_read.slang`). Every frame, `visPartFrames`
computes one affine map per part from the posed space back to the bind space
where the field lives:

$$x_{\text{rest}} = U A_{k_p}^{-1}\,(x' - t_{k_p}) - U g,\qquad U=L(G)^{-1},$$

where $k_p$ is the part's head joint, $X_{k_p}=(A_{k_p},t_{k_p})$ its skinning
transform and $g$ the translation of $G$; call this map $T_p^{-1}$. Before this map was precomputed it cost
two $3\times3$ inverses per gaussian per part per light, about 260 million a
frame on the sparrow; precomputing it took the read from 12.7 to 8.0 ms
(5 887 323 gaussians, 1600×1000). The factor for gaussian $i$ and light direction
$\omega_\ell$ is then the product over the other parts:

$$V_{i\ell}=\prod_{p\ne \pi(i)} F_p\!\left(T_p^{-1}x_i,\;L(T_p)^{-1}\omega_\ell\right),$$

where $\pi(i)$ is the gaussian's own part, $F_p$ is read trilinearly over the probes and bilinearly over the
octahedral map, and is 1 outside the part's box. A dome uses the ambient term
instead. The factors go into the same `shadowFactors` that both routes read,
so the traced route, which had no splat shadows at all, gets them too. No ray
is traced per frame.

**Found while building it.** An empty part's unwritten texels read as total
shadow and turned the bird black; a grid of 0 now marks it and it is skipped.
The bake traces inline rays, which the CUDA target lacks, so it is a separate
module and CUDA devices read fields baked elsewhere. A part does not shadow
itself: its field, read at its own gaussian, is the crowd it stands in. An
$8\times8$ map (about 22° per texel) cannot resolve a wing's edge; 16 and 32
are exact outside a band of one probe cell.

**Measured** on the sparrow: 4 269 858 gaussians, 609 joints, 12 parts,
grid 24, octave 16.

| | |
|---|---|
| bake (M5 Pro) | 22 s, 3 538 944 texels per part ($24^3\times16^2$) |
| fields in the file | 81 MB (two f16 per word) |
| factors under ½, frame 31, sun + dome | 21.3 % |
| viewer raster, 400 frames, median / warm | 71.82 / 9.69 ms (56.87 / 13.40 without) |
| viewer `rt`, median / warm | 193.88 / 51.65 ms (178.99 / 55.05 without) |
| L4 (CUDA) raster, 8 frames 1280×720 | 73.04 ms (67.48 without) |
| L4 (CUDA) `rt` | 128.77 ms (121.48 without) |

On the L4 the fields cost 5–7 ms per frame for
4.27 M gaussians × 2 lights × 11 parts of lookups. The shadow lands where the
mesh's does (the far wing, the flank under the raised wing, the tail root) and
follows the wings across frames. With every feather card included (9.5 M
gaussians at resolution 1400), the bake ran out of memory at the largest part
on both the M5 Pro (Metal) and an L4 through Vulkan (a 1.17 GB BLAS for about
3.3 M gaussians). The bird is therefore baked at resolution 1100.

**Limits.** A part's own local shadowing (feather on feather) is left out.
The ambient term is a mean over the sphere rather than a cosine lobe. Parts
are assumed rigid and to occlude independently. The floor under the bird is
shadowed by a separate light-space map, which was stabilised against the
"breathing" of a posed bounding box (sized from the rest sphere, snapped to
the world texel grid). A density-grid alternative (proposal 018) is recorded
but not implemented.

## 5.9 Levels of detail and decimation for skinned clouds

The `.athc` octree merges Morton-ordered cells. That cannot apply to a skinned
cloud, because a cell that took wing and body together would not know which
joint to follow. Two mechanisms replace it.

**Discrete levels by reconversion.** `mesh2splat --lod-levels N` converts at
the requested resolution, then at half of it, and so on, each level a cloud
with its own rig. `Engine::lodLevelsFor` draws the coarsest level whose cell
spans no more than `athenea:lod:threshold` pixels (default 1) at the nearest
point of its *posed* bounds. Undrawn levels are not posed (`deferredPose`).
Sparrow clips at 1100 / 550 / 275 cells (5.9 M / 1.5 M / 0.4 M gaussians),
1600×1000: 49.6 → 51.5 ms when the bird fills the frame; **71.3 → 24.7 ms** at
four times the distance. Every level stays resident with its own visibility
fields (81 MB each), and switching between levels is a step, not a blend.

**Decimation that keeps the rig.** `usd::decimateStage` finds every
per-gaussian array and merges each by kind (`lod_attributes.slang`). When
`skel:jointIndices` has a matching `skel:jointWeights`, the pair is merged as
a rig (mode 3). Over the run of records $R$ that a kept gaussian stands for,
each with merge weight $\omega_r$,

$$c_j=\sum_{r\in R}\omega_r\sum_k w_{rk}\,[\,j_{rk}=j\,],$$

collecting up to 16 distinct joints. The `width` heaviest are kept and
renormalised, $w'_j = c_j/\sum_{\text{kept}} c$. A record's dominant joint is
also folded into the merge key, so a merge never crosses a joint's edge.
Checked: a floor whose halves belong to two joints goes from 4096 to 256
gaussians, each kept gaussian wholly on its side's joint. The sparrow's finest
level (5 887 323 gaussians, with ids, material, visibility parts and rig)
keeps 87.7 % in 28 s. Posed at time 20, it draws as the undecimated cloud at
an image RMS of 0.002.

## 5.10 The sparrow

The sparrow and a BMW were the first two assets converted that were not
authored for this renderer, and most of what it uncovered was outside the skinning maths.

- **Animation export.** Blender's exporter wrote no joint samples for this
  FBX-imported rig, so `scripts/sparrow-anim.py` samples the evaluated pose.
  Solving a change of basis against `restTransforms` mangled the head, bill
  and toes, because rest and bind are different poses on this bird (worst
  1.989 at `Toe.L.010`). It now writes $W(t)=P(t)\,L^{-1}B$, so that
  $W(t)B^{-1}$ is Blender's own skinning matrix; what remains is
  quantisation, $4.9\times10^{-5}$–$7.1\times10^{-5}$ at frames 1, 31, 61.
- **Corrected stage.** `SparrowBird.usda` carries the asset fixes: no
  metallic on feathers or eyes, an unpremultiplied feather normal map
  (`UV2_treesparrow_normal_fixed.png`), and the second UV set (`UVMap_001`)
  for the feather alpha. Every later measurement uses it.
- **Cut-outs and budget.** Honouring the feathers' connected alpha `opacity`
  cut their budget from 15 378 695 to 5 898 220 gaussians at resolution 1400.
  Meshes past the 2 M-per-run ceiling are converted in slices; with every card
  the bird is 9.5 M gaussians.
- **Mesh reference.** Making the raster and path-traced mesh renders agree
  took the raster's relMSE against the tracer from 7.28 to 3.67 (960×540).

Measured cloud against mesh (`--no-bake`, wing camera, alpha without dome,
relMSE with it), after coverage was made to follow opacity:

| | alpha before → after (mesh) | relMSE before → after |
|---|---|---|
| sparrow wing, raster | 0.2395 → 0.2367 (0.2253) | 0.1742 → 0.1635 |
| sparrow wing, traced | 0.2385 → 0.2358 (0.2462) | 0.1331 → 0.1247 |

For the web, the bird was converted with `--skinned --transfer
--resolution 512`, with zonal lobes baked at 256 paths and 3 bounces in the
bind pose and the 12-part visibility added (`sparrow_tx_vis.usdc`: 1 504 160
gaussians, 1 504 096 with content). Against the mesh path traced at 1024
paths (1024², `air_fly_A0`, frame 16), the cloud's raster render reads
relMSE 0.0025. The cloud's body comes out slightly darker and browner,
because the zonal transfer keeps only direct light.

<!-- Figure: /Users/muriel/luc/athenea-pieces/docs/images/sparrow-mesh-cloud.png -->
<!-- Figure: /Users/muriel/luc/athenea-pieces/docs/images/sparrow-cloud.png -->
<!-- Figure: /Users/muriel/luc/athenea-pieces/docs/images/sparrow-film.png -->
<!-- Figure: /Users/muriel/luc/athenea-renders/sparrow-web/ref_pt_raster.png -->
<!-- Figure: /Users/muriel/luc/athenea-renders/sparrow/fixed_wing_athenea.png -->

## 5.11 The pawn: the rigid control

The chess pawn (The Open Chess Set, ASWF, CC BY 4.0) has **no skeleton**. It
is two meshes: a black-marble body `M_Pawn_Body_B` and a head `M_Pawn_Top_B`
with `transmission = 1`, a glass ball on a gold ring. It is in this section
because it is the asset every animation-related change had to leave
unchanged:

- When `MeshStageOptions` gained a time, the pawn converted to a file
  identical byte for byte to the one before.
- When the flat axis became $\min(\text{along}_U,\text{along}_V)\times0.1$ to
  fix the fox's ghosting under the ray tracer, the pawn's marble moved from
  0.0644/0.0798/0.0778 to 0.0645/0.0799/0.0780.
- Every skinned path, with identity transforms, reduces to the rigid one, and
  the pawn is the reference for the rigid one.

Its first end-to-end conversion produced 729 073 gaussians at resolution 512.
Its material and transfer results belong to Sections 3 and 4. The
high-quality white-pawn validation (`athenea-renders/pawn-hq`,
`pawn_mesh.usda`, 2048², 2048 reference paths) records, per material:

| material | gaussians | relMSE | p99 | mean (cloud) | mean (path-traced mesh) |
|---|---|---|---|---|---|
| `M_Pawn_Body_W` | 1 568 499 | 0.0218 | 0.273 | 0.167 / 0.165 / 0.133 | 0.164 / 0.162 / 0.132 |
| `M_Pawn_Top_W` (glass) | 307 296 | 0.1418 | 0.354 | 0.236 / 0.260 / 0.221 | 0.365 / 0.397 / 0.279 |

Three later runs at 1024² in the same folder (`s49`, `s54`, `s59`) read
glass relMSE 0.149, 0.081 and 0.043, with the body unchanged at 0.025. What
changed between them is not recorded next to them. The body is within a few
percent of the path-traced mesh. The error is concentrated in the glass head.

<!-- Figure: /Users/muriel/luc/athenea-renders/pawn-hq/pawn_gt_gs.png -->
<!-- Figure: /Users/muriel/luc/athenea-renders/pawn-hq/ring_gt_gs.png -->

## 5.12 Playback on the web and on a phone

The web player (`sparkwebGPU`, a WebGPU fork of Spark) reuses athenea's
kernels compiled to WGSL. athenea's `.athc` writer refuses `--skinned`, so the
web side reads the USD cloud (`usd-athc`, built on the `openusd` crate) and
writes a `.athc` v3. That file adds a `SKIN` section per element, with the
influences packed as joint u16 | weight unorm16 and the gradients as halves,
and an `ATSK` skeleton blob holding joint names, the bind transform and the
first clip. Every other clip is an `ATCL` file fetched when it is played.
Merged LoD nodes carry the opacity-weighted blend of their splats' influences
(the four heaviest, renormalised) and no gradient.

The skinning kernel (`athenea_adapter/skin.slang`) restates `splatSkin` line
for line, because the original binds eleven storage buffers and Chrome on a
Mac allows ten per stage. A GPU test runs athenea's own kernel on the same
splats and compares the two. Against athenea's raster the result is relMSE
0.0015–0.0053 with a brightness ratio of 1.01. No shutter motion is written.

| page | content | measured |
|---|---|---|
| `/pawn` (published 2026-10-06) | body 727 061 + head 673 711 splats, transfer, 6 CC0 HDRIs | 60 fps in Chrome; v8 data set: download 281.3 → 82.7 MB, not uploaded; desktop GPU memory 1127 → 800 MB |
| `/sparrow` (published 2026-10-07) | 1 504 096 splats, 125.7 MB, 12 clips (6.4 MB gzip), zonal transfer, HDRI | desktop GPU memory 1084 → 684 MB |
| `/sparrow` on phones | `sparrow-mobile`: 488 656 splats, 31.7 MB, cut from the LoD tree (`--keep-splats 600000`) | 60 fps headless in mobile mode, animated |
| sparrow-v2 (candidate) | 1 100 334 splats (−26.8 %), 75.9 MB; mobile 371 138, 26.1 MB | same error as published (4.46e-3) |

**Pruning a moving asset.** Gaussians hidden in the bind pose cannot simply be
dropped, because other poses reveal them. Measured over 12 clips × 4 samples
= 48 poses plus the bind pose, 61 views each: 61.2 % of the sparrow is hidden
in the bind pose, but only **32.5 %** is hidden in every pose. Removing what
the bind pose hides costs relMSE up to $4.5\times10^{-2}$ in close views of
unsampled poses. Removing what every pose hides costs at most
$2.9\times10^{-4}$. sparrow-v2 is the latter. For comparison, the static pawn
has only 2.7 % hidden.

## 5.13 What is not done

- More than four influences per gaussian (P014C). This limits accuracy at
  the sparrow's wing tips, both in the animation and in the bake's agreement
  with the mesh.
- Dual-quaternion skinning for clouds. Retargeting and clip blending, which a
  cloud holding baked transforms cannot do.
- Several skinned meshes at different prim transforms (only the first one's
  is kept), and skeletons with their own animated transform.
- The traced route poses once per frame, so it does not blur a skinned cloud.
  LoD instances carry no motion.
- The part's own self-shadowing term. Cross-part occlusion in poses other
  than the bind pose, inside the zonal transfer. Visibility bakes beyond about
  3 M gaussians per part.
- The zonal fit's error on the sparrow, and a geometric measure of coverage
  at a close bend, which is what would show the gain of the elastic Jacobian
  term.
