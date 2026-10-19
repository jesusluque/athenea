<!-- Copyright (c) 2026 jesus luque. -->

# 5. Animated assets: skeletons, skinning and per-part visibility (the pawn and the sparrow)

This section describes how athenea turns a rigged USD asset (UsdSkel) into a
cloud of 3D Gaussians that a skeleton still drives. It covers how the cloud is
converted, how it is posed on the GPU every frame, how its light and
self-shadowing are kept valid under deformation, how levels of detail treat it,
and, as an application, how it is played back in a browser and on a phone. Two assets run through
the section. The Eurasian tree sparrow (`SparrowBird.usda`, 609 joints, 71
animation clips) is the animated test case. The chess pawn of The Open Chess
Set is the rigid control. Every number below is taken from
`docs/decisions.md`, the source files named alongside it, or the measurement
files of the web player (`sparkwebGPU`). Numbers that were not measured are
marked as not measured.

::: {.nota}
**UsdSkel** — the OpenUSD domain [43] that describes articulated characters. Its pieces: a `SkelRoot` prim, which bounds the subtree where skinning is resolved (nothing outside a `SkelRoot` deforms); a `Skeleton` prim, with the `joints` array (tokens that are paths, e.g. `Root/Spine/Chest`, so the hierarchy is encoded in the name itself rather than in child prims), `bindTransforms` (each joint's world transform at the moment the mesh was bound) and `restTransforms` (the *local* transforms of a rest pose, which need not match the bind pose); a `SkelAnimation` prim, with per-joint `translations`, `rotations` and `scales` (local, sampled in time) and `blendShapeWeights`; and the applied schema `SkelBindingAPI`, which binds a geometry to a skeleton through `primvars:skel:jointIndices`, `primvars:skel:jointWeights`, `primvars:skel:geomBindTransform`, `skel:joints`, `skel:skeleton` and `skel:skinningMethod`.

**Rig, skeleton, joint** — a *rig* is the set of controls that deforms a model; in UsdSkel it reduces to a *skeleton*: a tree of *joints* ("bones"), each with a transform relative to its parent. Composing the local transforms from the root gives each joint's world transform at each instant.

**Blend shapes** (morph targets) — UsdSkel's other form of deformation: per-vertex offsets added to the rest geometry, weighted by `blendShapeWeights`. They are used for faces and fine corrections. The engine applies them to meshes (`geom::Skinner`, `shaders/athenea/geom/skinning.slang`, `skinPoints`), but this section does not apply them to clouds: a cloud carries joint skinning only.

**The two assets** — the sparrow is an *animated* control with 609 joints (a high count: every wing and tail feather has its own chain of joints), which tests the rig's scale; the pawn has no skeleton and serves as the *rigid* control: any change to the animation path must leave it unchanged.

*References:* [43] in §10; UsdSkel specification: openusd.org/release/api/usd_skel_page_front.html; code: `modules/usd/src/MeshStage.cpp`, `modules/geom/include/athenea/geom/Skinner.h`; `docs/decisions.md` (sections "Skinning and blend shapes, on the device", "A cloud carried by a skeleton, on the device").
:::

## 5.1 Design principle: carry the rig, not the frames

There are two ways to animate a Gaussian cloud. The first stores every
gaussian's position, orientation and scale at every frame. The second stores
the gaussians once, in the bind pose, together with the joints that carry
each one, plus one transform per joint per frame. athenea uses the second, for
three reasons, each measured:

::: {.nota}
**Bind pose** — the skeleton pose in which the geometry was bound: the mesh is modelled in that pose, and a joint's skinning matrix, $X_j(t)=W_j(t)\,B_j^{-1}$ (the joint's world transform at $t$ times the inverse of its bind transform), is the identity there. That is why "storing the gaussians in the bind pose" means storing them in the space where the skinning matrices apply directly.

**The two alternatives** — (a) a *vertex/point cache*: every frame is a complete cloud (what an Alembic point cache or one PLY per frame would do, or the 4D-GS methods that store or predict attributes per instant); (b) a *rig*: one static cloud plus a parametric deformation function (linear blend skinning) driven by a few matrices per frame. It is the same choice production faces between exporting a geometry cache and exporting the rig and animation; here (b) is chosen for the three reasons that follow.

*References:* UsdSkel, "Schemas: Skeleton, bindTransforms"; code: `shaders/athenea/scene/splat_skin.slang` (header, "What it is for"); `docs/decisions.md` (section "The file carries the rig, not the frames").
:::

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

::: {.nota}
**The size figures** — with $150\,000$ gaussians and 120 frames there are $1.8\times10^7$ (gaussian, frame) pairs. The rig's cost fits u16 indices and f16 weights: $4\times(2+2)=16$ B per gaussian gives $2.4$ MB, and $60\times120$ $4\times4$ float matrices ($64$ B) give $0.46$ MB: in total $\approx2.9$ MB. The 960 MB / 1.2 GB of the per-frame arrays are the estimate in `docs/decisions.md`, which does not detail which attributes it counts; the measured tube figure, on the other hand, can be reconstructed: $713\ \text{MB}/(184\,320\times121)\approx32$ B per gaussian per frame, about 16 halves (position, rotation, scale and padding). "Five times the frames cost 230 kB": going from 25 to 121 instants adds 96 frames $\times$ 40 joints $\times$ 64 B $=245\,760$ B $\approx 240$ kB, in line with the 230 kB measured.

**Exactness between samples** — LBS is a continuous function of the joint matrices; if `skinningXforms` is sampled in time, USD interpolates the matrices linearly between samples (or the reader can ask for any instant of the `SkelAnimation`), and the cloud is evaluated *exactly* with those matrices. A per-frame cache forces positions and, above all, rotations and covariances to be interpolated gaussian by gaussian, with artefacts (shrinkage when rotations are interpolated linearly, order jumps if the cloud changes).

**Stable identity and refit** — a BVH (bounding volume hierarchy) built over the cloud keeps its topology if the number and order of the primitives do not change; recomputing the boxes bottom-up (*refit*) is enough, and much cheaper than a rebuild (Morton sort, hierarchy construction). The technique is that of Wald et al. (2007), "Ray Tracing Deformable Scenes Using Dynamic Bounding Volume Hierarchies", and Lauterbach et al. (2006), "RT-DEFORM".

*References:* [32] in §10 (Karras's LBVH); code: `modules/render/src/GaussianRayTracer.cpp` (`GaussianRayTracer::sync`); `docs/decisions.md` (sections "The file carries the rig, not the frames", "What a rigged cloud actually costs", "A posed cloud refits its ray tracing structure").
:::

The cost of this choice is that whatever was baked in world space (radiance,
spherical-harmonic transfer, visibility) stops being valid as soon as a limb
moves. §5.7 and §5.8 deal with this.

::: {.nota}
**Baking in world space** — baked radiance (colour already lit), PRT transfer in 9 spherical harmonics (bands $l=0..2$, $9=\sum_{l=0}^{2}(2l+1)$ coefficients) and visibility are expressed in world coordinates. If a wing turns $90^\circ$, its gaussians still "see" the sky and the occlusions of the conversion pose: the light stays attached to space, not to the limb. The two solutions that follow split the problem in two: what turns *with* the gaussian (transfer in its local frame, §5.7) and what depends on the relative position between limbs (per-part visibility, §5.8).

*References:* [14] (PRT, Sloan, Kautz, Snyder 2002) and [17] in §10; `docs/decisions.md` (section "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame").
:::

## 5.2 Reading a rigged stage

The converter (`athenea mesh2splat`) does not go through Hydra. It reads
meshes through its own `MeshStage`, which leaves two ways to handle a skinned
mesh:

- **Posed.** Without `--skinned`, the stage is posed before it is read.
  `UsdSkelBakeSkinning` writes the posed points as time samples into an
  in-memory override layer above the root layer, over `GfInterval(t, t)`. This costs one pose, leaves the
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

::: {.nota}
**`MeshStage`, and why not Hydra** — the converter reads `UsdGeomMesh` directly with its own reader (`modules/usd/src/MeshStage.cpp`, `modules/usd/include/athenea/usd/MeshStage.h`). Hydra would resolve the skinning (usdSkelImaging turns it into ext computations), but the converter needs the geometry in a specific pose, not a render frame.

**`UsdSkelBakeSkinning`** — a UsdSkel utility that evaluates skinning and writes the deformed points as time samples on the meshes themselves. The `poseStage` function calls it inside a `UsdEditContext` on the stage's **session layer** (the "in-memory override layer": the strongest layer, never saved to disk) and over the degenerate interval `GfInterval(t, t)`, so only one instant is computed. If the stage has no `SkelRoot`, the call does nothing.

**`UsdSkelCache`, `UsdSkelSkinningQuery`, skel binding** — `UsdSkelCache` is populated by walking each `SkelRoot` and returns, for every prim with `SkelBindingAPI`, a `UsdSkelSkinningQuery`: the object that knows how to compute the influences (`ComputeVaryingJointInfluences`), their number per point (`GetNumInfluencesPerComponent`), the `geomBindTransform` (`GetGeomBindTransform`) and the method (`GetSkinningMethod`). The **geomBindTransform** is the mesh's world transform at bind time: it takes points from mesh space to the bind space in which the skinning matrices act.

**Why the bind pose with `--skinned`** — the engine's skinner applies $X_j$ to bind-pose points. If the converter read points already posed at $t$, the engine would apply $X_j(t)$ on top of them and the deformation would be applied twice.

**The square test** — a joint translating by $(1.2,0.4,0)$ between $t=0$ and $t=1$ moves everything it carries rigidly; the bounds (AABB) of the converted cloud at $t=0$ and $t=1$, folded by a reduction on the GPU, must differ by exactly that vector. It checks the posed path without comparing images.

*References:* [43] in §10; API: `UsdSkelBakeSkinning`, `UsdSkelCache`, `UsdSkelSkinningQuery` (openusd.org/release/api/usd_skel_page_front.html); code: `modules/usd/src/MeshStage.cpp` (`poseStage`, `skinningOf`, `resolveSkinning`), `tests/usd/test_usd.cpp`; `docs/decisions.md` (sections "The conversion reads a pose, not a rest", "A gaussian carries the joints its triangle carries").
:::

**Joint order.** A mesh may list its own subset of joints in `skel:joints`.
UsdSkel's mapper only maps skeleton order to mesh order, so it is run over the
identity to learn the skeleton index of each mesh index. Influences are always
written in skeleton order, so one cloud has one joint order no matter how many
meshes it was converted from. A mesh in a skinned stage that no joint carries
(the sparrow stage holds a cylinder and a plane beside the bird) still gets
one influence record per gaussian: four joints of zero weight, which the
skinner reads as "stay in the bind pose". This keeps rig and cloud aligned
one to one.

::: {.nota}
**`skel:joints` and the mapper** — a mesh may declare its own list of joints (a subset of the skeleton, in a different order), and then its `jointIndices` index that local list. UsdSkel provides a `UsdSkelAnimMapper` that translates arrays *from skeleton order to mesh order*. To obtain the inverse direction (local index → skeleton index), `skinningOf` remaps the identity array $[0,1,\dots,n-1]$: the result at local position $i$ is that joint's skeleton index. It is a way of reusing the API without reimplementing the name resolution.

**One order per cloud** — since a cloud may come from many meshes, each with its own subset, always writing in skeleton order guarantees that an index $j$ in the cloud always means the same thing and that `skinningXforms` (one matrix per skeleton joint) is indexed directly.

**Zero weights = bind pose** — in `splat_skin.slang`, if the weights sum to $\le 10^{-5}$ the gaussian is treated as "carried by nothing" and stays at $G\,p$, transformed by the same world matrices. Keeping one record per gaussian even when it does not move avoids sparse arrays and keeps the 1:1 correspondence between cloud and rig.

*References:* API `UsdSkelAnimMapper::Remap`; code: `modules/usd/src/MeshStage.cpp` (`skinningOf`), `shaders/athenea/scene/splat_skin.slang` (`splatSkin`, branch `!(total > kEps)`); `docs/decisions.md` (section "The conversion's order is the mesh's, and a slot with nothing in it is kept").
:::

**`--skinned` forces `--no-bake`.** A radiance bake holds the scene's
environment and bounces (the ground under a foot), and carrying that up with
a lifted limb amounts to rotating a lightmap. A skinned cloud carries its
material and is relit every frame. Its light transfer, when one is asked
for, is kept in a form that rotates with the gaussian (§5.7).

::: {.nota}
**Why `--skinned` implies `--no-bake`** — the radiance bake of §3–4 writes into each gaussian the light reaching it in the scene *as it is*: the ground bouncing light up to the leg, the visible sky, another object's shadow. It is a *lightmap* in the classic sense. When the limb is raised, that light travels with the gaussian even though its surroundings have changed: rotating a lightmap does not make it correct. That is why a skinned cloud keeps its material parameters (albedo, roughness, metallic…) and is *relit* every frame with the current lights. Only the part that depends on the cloud's own geometry (the self-shadowing transfer) is stored, and in a form that turns with the gaussian.

*References:* [14] in §10; code: `apps/athenea/src/CmdMesh2Splat.cpp` (options `--skinned`, `--transfer`); `docs/decisions.md` (section "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame").
:::

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

::: {.nota}
**Skinning weights (influence weights)** — each vertex has a list of (joint, weight) pairs with non-negative weights that sum to one; they describe how much the vertex follows each joint. In UsdSkel the number per vertex is the `elementSize` of `primvars:skel:jointIndices`/`jointWeights` (25 on the sparrow's wing).

**Barycentric coordinates** — a point $x$ of the triangle $(a,b,c)$ is written $x=\lambda_a a+\lambda_b b+\lambda_c c$ with $\lambda_a+\lambda_b+\lambda_c=1$. Interpolating the corner weights with them is linear interpolation over the triangle, exactly what rasterising a per-vertex attribute would do. So a gaussian's weight matches the one the mesh's surface point would have at that place.

**The reduction from twelve to four** — each corner contributes up to four pairs (`mesh_pack.slang` already kept the four heaviest per corner), i.e. twelve candidates. `blendInfluences` walks them: if the joint is already in a slot, it adds; if there is a free slot (weight 0), it takes it; otherwise it replaces the lightest only if the candidate weighs more. This selection is greedy and order-dependent; that is why, at the end, each kept weight is **recomputed** as its full blend $\sum_c w_{cj}\lambda_c$ (not what was accumulated since it entered the slot), and then renormalised: $w_k=\tilde w_k/W$. Renormalisation redistributes the weight of the discarded joints proportionally, so the gaussian moves "whole" even though it has lost influences (the resulting error is one of *position*; see Truncation).

**Why "the same layout as the mesh"** — the influence image has one entry per triangle corner, aligned with the triangle image the AOFX effect reads; so the effect addresses both with the same index, without auxiliary tables.

*References:* [12] in §10 (EA SEED's mesh2splat); code: `shaders/athenea/usd/mesh_pack.slang` (field `perPoint`, buffer `influences`), `plugins/mesh2splat/mesh2splat.slang` (`blendInfluences`); `docs/decisions.md` (section "A gaussian carries the joints its triangle carries").
:::

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

::: {.nota}
**Why gradients are needed** — the covariance of a deformed gaussian depends on the derivative of the deformation map (§5.4.2), and that derivative includes how the weights change in space. A constant weight per gaussian (the usual choice in Gaussian avatars that apply LBS) ignores that term.

**Gradient of the barycentrics** — with $e_1=b-a$, $e_2=c-a$ and $n=e_1\times e_2$ (unnormalised normal, $|n|=2\cdot$area), $\lambda_b$ and $\lambda_c$ are linear functions whose gradients in the triangle's plane are $\nabla\lambda_b=(e_2\times n)/|n|^2$ and $\nabla\lambda_c=(n\times e_1)/|n|^2$. Check: $\nabla\lambda_b\cdot e_1=(e_2\times n)\cdot e_1/|n|^2=n\cdot(e_1\times e_2)/|n|^2=1$ and $\nabla\lambda_b\cdot e_2=0$, as it should be (moving along $e_1$ increases $\lambda_b$ by 1). Since they sum to one, $\nabla\lambda_a=-\nabla\lambda_b-\nabla\lambda_c$. The code guards degenerate triangles (if $|n|^2\le10^{-30}$, the gradient is 0).

**Quotient rule** — $w_k=\tilde w_k/W$ with $W=\sum_{i\in K}\tilde w_i$, so $\nabla w_k=(\nabla\tilde w_k\,W-\tilde w_k\nabla W)/W^2=(\nabla\tilde w_k-w_k\nabla W)/W$. It is exact at the point evaluated (the gaussian's centre) but not constant: $w_k$ is a quotient of linear functions, hence not linear over the triangle.

**What is stored, and why 12 bytes** — (1) Only the tangential components: a conversion gaussian is an almost flat disc (minimal third axis), so only the derivatives along its rest axes $\hat u,\hat v$ matter: $(\nabla w_k\cdot\hat u,\ \nabla w_k\cdot\hat v)$. (2) Since $\sum_k w_k=1$, $\sum_k\nabla w_k=0$: storing three gradients is enough, and the fourth is $-(g_1+g_2+g_3)$. Three gradients $\times$ two components $=6$ halves $=12$ bytes (`elementSize` 6). The halves are clamped to $\pm65504$ (`m2sPackHalves`) so that a very abrupt transition does not become infinity.

**Nine entries** — the record the AOFX effect returns per gaussian (one float4 per entry) grows from eight to nine: the ninth holds the packed gradients. The AOFX SDK does not change; the host reads the record length from the effect itself.

*References:* standard vector calculus of linear shape functions (P1 finite elements); code: `plugins/mesh2splat/mesh2splat.slang` (`blendInfluences`, `m2sPackHalves`), `shaders/athenea/scene/splat_skin.slang` (reading of `weightGradients`, `lastGradient`); `docs/decisions.md` (section "The whole Jacobian of a skinned gaussian", paragraphs "What a gaussian keeps" and "Where it comes from").
:::

**Measured.** A square bound entirely to one joint converts to 1056
gaussians, all naming that joint with full weight. Through the effect, 272
gaussians of a quad whose corners all name joint 3 come out on joint 3, with
the other slots empty and the weights summing to one. With one joint
everywhere, the stored gradient is below $10^{-3}$.

::: {.nota}
**What these measurements prove** — they are invariant tests, not tests of visual quality: (1) a mesh bound to a single joint with weight 1 must give gaussians with that joint and weight 1 (the reduction and renormalisation do not invent influences); (2) if all three corners name the same joint, the blend must collapse to one slot with total weight 1 and three empty slots; (3) a constant weight has zero gradient, and the stored value must be rounding noise ($<10^{-3}$, on the order of half precision relative to the triangle's scale). The 1056 and 272 gaussians are simply what the test resolution produces for that geometry.

*References:* code: the mesh2splat tests in `tests/aofx/` and `tests/usd/test_usd.cpp`; `docs/decisions.md` (sections "A gaussian carries the joints its triangle carries", "The whole Jacobian of a skinned gaussian").
:::

**Truncation.** Four influences is a truncation. On the sparrow, 27 % of wing
vertices carry more than four (6686 vertices, `elementSize` 25), and a
gaussian takes the union of three corners' joints, so the share of truncated
gaussians is at least as large. That share has not been measured.

::: {.nota}
**Truncation to four influences** — this is the convention of real-time engines (four pairs fit in two vec4 registers, and most game rigs are limited to 4). Production rigs use many more: the sparrow's wing arrives with `elementSize` 25. Discarding influences and renormalising moves the point away from where the mesh, with all of its own, would put it: it is an error of **position** (and of frame), largest where many joints with similar weights compete, as at the feather tips. Since a gaussian merges the joints of three corners (up to 12 distinct ones), the proportion of truncated gaussians is at least that of vertices with more than four (27 %).

**A consequence measured elsewhere** — in §5.7 this same truncation made the bake's ray miss the mesh surface (566 603 transparent gaussians) when the bake was done in an animated pose.

*References:* `docs/decisions.md` (sections "The whole Jacobian of a skinned gaussian", paragraph "Not done here", and "A skinned transfer is traced at the bind pose", which mentions the case of more than four influences).
:::

## 5.4 Skinning a Gaussian on the device

`scene::SplatSkinner` dispatches `shaders/athenea/scene/splat_skin.slang`, one
thread per gaussian. Its inputs are the rest cloud, the (joint, weight)
pairs, the joints' skinning transforms at the current instant, and three
matrices: the bind transform $G$, `skelToWorld` and `worldToPrim`. USD
matrices are row-major and act on row vectors, so the host transposes them as
`geom::Skinner` does for meshes (Figure 14).

::: {.nota}
**`scene::SplatSkinner`** — the C++ class that prepares and dispatches the kernel (`modules/scene/include/athenea/scene/SplatSkinner.h`, `modules/scene/src/SplatSkinner.cpp`, `SplatSkinner::skin`). One thread per gaussian (`[numthreads(256,1,1)]`): there are no dependencies between gaussians; it is a pure map.

**The three matrices** — $G$ (`geomBind`): cloud space → the skeleton's bind space. Note a subtlety: the conversion writes the gaussians in world space, whereas UsdSkel's `geomBindTransform` starts from mesh space; `skinningOf` composes the two once, `toWorld^{-1}·geomBindTransform`, so that the file carries the single matrix a renderer needs. `skelToWorld`: from skeleton space to the world. `worldToPrim`: from the world to the cloud prim's local space, because the renderer expects the cloud in its own space and applies its prim transform afterwards.

**Row-major and row vectors** — USD (Gf) writes matrices to multiply row vectors on the left ($v' = v\,M$, translation in the last row). The kernels multiply a matrix by a column vector (`mul(m, v)`), so the host transposes (`transposed` in `SplatSkinner.cpp`; likewise in `Engine::carryCloud` for the joint matrices), just as `geom::Skinner` does for meshes.

*References:* code: `modules/scene/src/SplatSkinner.cpp` (`transposed`, `setRows`), `modules/usd/src/MeshStage.cpp` (`skinningOf`, comment "OUT OF THE CLOUD'S SPACE"), `modules/usd/src/Engine.cpp` (`Engine::carryCloud`); `docs/decisions.md` (sections "A cloud carried by a skeleton, on the device", "And the engine puts the cloud where the skeleton is", paragraph "Where the spaces meet").
:::

![Figure 14. Skinning on the device: the bind-pose cloud, the per-Gaussian joint weights and their gradients, and the joint transforms of the current time give the posed mean and covariance; per-part visibility tables baked once in the bind pose shadow the posed cloud.](figures/fig-skinning.svg)

### 5.4.1 The mean

Let $q = G\,p$ be the rest point in bind space, $X_j q = A_j q + t_j$ joint
$j$'s skinning transform, and $M=\texttt{worldToPrim}\cdot\texttt{skelToWorld}$.
The posed mean is linear blend skinning, with exactly the arithmetic the mesh
skinner uses:

$$\mu' = M \sum_j w_j\,X_j\,(G\,\mu).$$

::: {.nota}
**Linear blend skinning (LBS)** — also called *skeletal subspace deformation* (Magnenat-Thalmann et al., 1988): the deformed point is the convex combination of the points each joint would give on its own, $p'=\sum_j w_j X_j q$. With $X_j=(A_j,t_j)$ affine, $\sum_j w_jX_j$ is another affine matrix: the *matrices* are blended, not the points, which is why the kernel accumulates a single `blended` matrix that serves for the position and (its linear part) for the frame.

**Skinning matrix** — in UsdSkel, `ComputeSkinningTransforms` returns $X_j(t)=W_j(t)\,B_j^{-1}$ in skeleton space: first the joint's bind transform is undone, then the current one is applied. In the bind pose $X_j=I$.

**LBS's known defect** — blending rotations linearly does not give a rotation: near a $180^\circ$ twist the 50 % blend collapses (*candy wrapper*), and strong bends lose volume. The alternatives are **dual quaternion skinning** (Kavan, Collins, Žára, O'Sullivan, 2007/2008: it blends dual quaternions, which do represent rigid transforms, and normalises), *optimized centers of rotation* (Le and Hodgins, 2016) or blend-shape corrections (Pose Space Deformation, Lewis et al., 2000). athenea chooses LBS because it is exactly what the mesh skinner does (so that cloud and mesh match) and because its derivative is simple (§5.4.2). DQS exists for meshes (`SkinningMethod::DualQuaternion` in `geom::Skinner`), not for clouds.

*References:* Magnenat-Thalmann, Laperrière, Thalmann (1988), "Joint-dependent local deformations for hand animation and object grasping"; Kavan et al. (2008), "Geometric Skinning with Approximate Dual Quaternion Blending", ACM TOG; code: `shaders/athenea/scene/splat_skin.slang` (`splatSkin`, `moved`, `blended`), `shaders/athenea/geom/skinning.slang`; `docs/decisions.md` (section "Skinning and blend shapes, on the device").
:::

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

::: {.nota}
**Jacobian of the deformation** — if $\phi(p)$ is the position map, the rest gaussian $\mathcal N(\mu,\Sigma)$ is approximated after deformation by the local linearisation $\phi(x)\approx\phi(\mu)+J(x-\mu)$, with $J=\partial\phi/\partial p$ at $\mu$; the image of a gaussian under an affine map is another gaussian with covariance $J\Sigma J^{\mathsf T}$. It is the same local affine approximation as the EWA splatting of Zwicker et al. [3], which 3DGS [1] uses to project to the screen ($\Sigma_{2D}=JW\Sigma W^{\mathsf T}J^{\mathsf T}$).

**Derivation** — $\phi(p)=M\sum_j w_j(p)\,X_j(Gp)$. By the product rule: $\partial\phi/\partial p=L(M)\big[\sum_j w_j\,A_j\,L(G)+\sum_j (X_jq)\,(\nabla_p w_j)^{\mathsf T}\big]$; the first summand differentiates $X_j(Gp)$ with the weight fixed (linear part $A_jL(G)$); the second differentiates the weight with the point fixed (the outer product of a column vector, the point carried by the joint, with the row gradient). $L(\cdot)$ takes the $3\times3$ block of an affine matrix.

**The elastic term, intuitively** — if two joints send the same point to different places, the gaussian lying in the weight transition is *stretched* between both destinations, like a rubber band. Estimate: with two joints, $\nabla w_2=-\nabla w_1$ and $|\nabla w|\approx1/L$; the term is $(X_2q-X_1q)\nabla w_2^{\mathsf T}$. If the joints differ by a turn $\theta$ about a pivot at distance $d$, $|X_2q-X_1q|=2d\sin(\theta/2)$; against the first term (of norm $\approx1$) the ratio is $\sim 2d\sin(\theta/2)/L$, which the text writes with $\tan$ (the same order at moderate angles). With $\theta=90^\circ$ and $d\approx L$, $\approx1.4$: "of order one". It is not a minor correction at elbows, wrists or the wing's hand.

*References:* [1], [3] in §10; code: `shaders/athenea/scene/splat_skin.slang` (header "THE JACOBIAN, WHOLE"); `docs/decisions.md` (section "The whole Jacobian of a skinned gaussian").
:::

The kernel builds the elastic term in the same loop that forms $X_k q$. It
accumulates $\sum_k (X_k q)\,g_k$ on each rest axis, *relative to the blended
point*. The gradients sum to zero, so this is the same sum, but it is computed
as a difference of nearby points, which stays exact in float far from the
origin. A joint with zero weight but a non-zero gradient still contributes.

::: {.nota}
**How it is implemented at no extra cost** — in the per-joint loop the kernel already computes `carried` $=X_kq$; with the gradient $g_k=(g_{k,u},g_{k,v})$ it reads, it accumulates `alongU` $\mathrel{+}= X_kq\,g_{k,u}$ and `alongV` likewise. These are the columns of the elastic term applied to $\hat u$ and $\hat v$ (the full $3\times3$ matrix is not needed, only its action on the disc's two axes).

**Relative to the blended point (numerical cancellation)** — since $\sum_kg_k=0$, $\sum_k(X_kq)g_k=\sum_k(X_kq-y)g_k$ for any $y$. If the cloud is far from the origin (e.g. coordinates of 100 m), the $X_kq$ are large numbers whose combination with weights of opposite sign cancels catastrophically in float32 (about 7 digits). Subtracting $y=\mu_{\text{blended}}$ makes each term a difference of nearby points, and precision is kept. Since the gradients are read as halves, their sum is not exactly zero; the code subtracts $y\cdot$`gradientSum`, which makes the identity exact for the values read.

**Zero-weight joints with a gradient** — at the edge where a corner's contribution vanishes, the weight is 0 but its derivative is not: the joint "pulls" even though it does not "carry". The code handles it (`pulls`).

*References:* Goldberg (1991), "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (cancellation); code: `shaders/athenea/scene/splat_skin.slang` (`alongU`, `alongV`, `gradientSum`, `pulls`).
:::

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

::: {.nota}
**Covariance of a disc** — a conversion gaussian is almost 2D (as in 2DGS [5] or surfels [6]): $\Sigma=R\,S^2R^{\mathsf T}$ with axes $\hat u,\hat v,\hat n$, scales $s_u,s_v$ and a minimal thickness. Restricted to the plane, $\Sigma_\parallel=E S^2E^{\mathsf T}$ with $E=[\hat u\ \hat v]$ ($3\times2$). The posed one is $\Sigma'=JES^2E^{\mathsf T}J^{\mathsf T}=[a\ b][a\ b]^{\mathsf T}$, with $a=J\hat u\,s_u$ and $b=J\hat v\,s_v$, the "carried" axes.

**Why not Gram–Schmidt** — the earlier version orthonormalised $a,b$ keeping the direction of $a$: that throws away the *shear* (a rhombus becomes a rectangle with the sides of $a$), and the elastic term is mostly shear. Diagonalising $\Sigma'$ gives the correct ellipse: axes = eigenvectors, sizes = square roots of the eigenvalues.

**Closed-form $2\times2$ diagonalisation** — take the orthonormal basis $e_1=a/|a|$, $e_2\propto b-(e_1\cdot b)e_1$; in it $[a\ b]=[e_1\ e_2]K$ with $K=\begin{pmatrix}k_{00}&k_{01}\\0&k_{11}\end{pmatrix}$ (a QR factorisation). $C=KK^{\mathsf T}$ is symmetric: $c_{00}=k_{00}^2+k_{01}^2$, $c_{01}=k_{01}k_{11}$, $c_{11}=k_{11}^2$. Its eigenvalues are $\lambda_\pm=m\pm r$. Since $\lambda_+\lambda_-=\det C=(k_{00}k_{11})^2$, the smaller one is computed as $\det C/\lambda_+$: for a thin gaussian ($\lambda_-\ll\lambda_+$), $m-r$ subtracts two nearly equal numbers and gives 0 or a negative in float; the quotient keeps all the significant digits. The angle is $\tfrac12\operatorname{atan2}(2c_{01},c_{00}-c_{11})$.

**Identity stability** — if $|c_{01}|$ is only rounding (a rigid pose, a nearly round gaussian), the basis $(e_1,e_2)$ is kept: without this a round gaussian would turn arbitrarily from frame to frame (the eigenvector of a double eigenvalue is undetermined). In addition, the eigenvector closest to $a$ is chosen as the first axis (swapping sizes if the angle exceeds $45^\circ$), so as not to permute axes between frames.

**The normal through the cofactor** — for vectors $a,b$ and an invertible matrix $J$, $(Ja)\times(Jb)=\det(J)\,J^{-\mathsf T}(a\times b)=\operatorname{cof}(J)(a\times b)$. It is the transformation rule for normals (the "inverse transpose" of raster engines), written without inverting: the kernel builds two tangents of the shading normal, carries them with the full $J$ and takes their cross product.

*References:* [1], [5], [6] in §10; Golub and Van Loan, *Matrix Computations* (QR and symmetric $2\times2$ problems); code: `shaders/athenea/scene/splat_skin.slang` (block "THE FRAME", `k00`…`minor`, `angle`, block "THE SHADING NORMAL"); `docs/decisions.md` (section "The whole Jacobian of a skinned gaussian", paragraph "The frame, exact"; section "A gaussian keeps its shading normal apart from its frame").
:::

The kernel writes only position, rotation and the two in-plane scales.
Opacity, colour, harmonics, PBR channels and the word holding the third size
are left untouched. A frame therefore costs one kernel over the cloud and
nothing is decoded again.

::: {.nota}
**What is rewritten** — the `GpuSplats` layout packs four shape words per gaussian (`shape`): [0] a smallest-three quaternion (10+10+10 bits and the dropped index), [1] $\log s_x$ and $\log s_y$ in f16, [2] $\log s_z$ and the base colour's red, [3] the base colour's green and blue. The kernel rewrites word 0 (rotation, `encodeQuaternion`) and word 1 (log of $s_u,s_v$), copies words 2 and 3 as they are, and writes `positions`. Opacity (in `positions.w`), SH and the PBR channels do not depend on the pose. That is why `Engine::carryCloud` creates a posed `GpuSplats` that **shares** the harmonics and PBR buffers with the rest cloud: the per-frame cost is a single dispatch over the cloud, with nothing re-decoded or uploaded.

*References:* code: `shaders/athenea/common/packing.slang` (layout of `shape`), `shaders/athenea/scene/splat_skin.slang` (`shape[row]`…`shape[row + 3]`), `modules/usd/src/Engine.cpp` (`Engine::carryCloud`).
:::

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

::: {.nota}
**The strip test** — 4096 gaussians on a strip with a weight that grows linearly ($w$ from 0 to 1 along the strip), two joints: one still ($I$) and one that turns a quarter turn ($R$). For this case the Jacobian has a closed form: $\phi(p)=(1-w)p+wRp$, so $J=(1-w)I+wR+(Rp-p)\nabla w^{\mathsf T}$, which is the general formula with $X_1=I$, $X_2=R$ and $\nabla w_2=-\nabla w_1=\nabla w$. The *oracle* is analytic and evaluated in a test kernel (`shaders/athenea/test/splat_skin_check.slang`), not on the CPU, and only a counter is read.

**Relative Frobenius error** — $\|\Sigma'_{\text{kernel}}-\Sigma'_{\text{analytic}}\|_F/\|\Sigma'_{\text{analytic}}\|_F$, with $\|A\|_F=\sqrt{\sum_{ij}a_{ij}^2}$. Threshold 3 %.

**Why 3 %** — the quaternion is stored as a 10-bit *smallest-three*: the component of largest magnitude is dropped (recoverable by normalisation) and the other three, bounded to $[-1/\sqrt2,1/\sqrt2]$, are quantised to 1024 levels. The step is $\sqrt2/1023\approx1.4\times10^{-3}$ in quaternion components, on the order of a tenth of a degree in the axis. Together with the half-precision scales, a 3 % threshold separates the format's error from the method's.

**Reading the table** — "before" is the earlier kernel (first term only, and Gram–Schmidt): it fails on half the strip (2048), exactly where the weight ramp and the bend create shear; it fails on the whole strip with a shearing joint and constant weights, because Gram–Schmidt removes the shear even when there is no elastic term. "After" fails in no case with gradients. The rigid-control row shows that neither breaks the trivial case. The last row (new kernel, file without gradients) isolates each piece's contribution: the proper frame fixes the shear, but without the elastic term 1984 of 4096 still fail.

*References:* code: `tests/scene/test_loading.cpp` (case "a skinned gaussian follows the whole Jacobian of its blend"), `shaders/athenea/common/packing.slang` (header: "rotation in 10-bit smallest-three"); `docs/decisions.md` (section "The whole Jacobian of a skinned gaussian", paragraph "Measured").
:::

An earlier test placed 4096 gaussians on a helix, each with its own frame and
size. At rest every gaussian stays where it was. Under a rigid turn of 1.2 rad
about a slanted axis plus a slide, every gaussian is carried rigidly. Only
four counters are read back.

::: {.nota}
**The helix** — a regression test from the initial version: gaussians with different frames and sizes (a helix guarantees each has a different orientation) under a rigid transform (a turn of 1.2 rad about a tilted axis plus a translation). It must hold that $\mu'=R\mu+t$, $R'=R\,R_{\text{rest}}$, and the scales are unchanged. Four counters are read (e.g. gaussians out of tolerance in position, rotation, scale, and at rest), following the rule of not reading data back to the CPU.

*References:* code: `tests/scene/test_loading.cpp` (case "a skinned cloud follows its joints, and nothing else moves"); the repository's `CLAUDE.md`, rule "No CPU arithmetic on data".
:::

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

::: {.nota}
**The sparrow figures** — on the `Flight` clip, for 1500 wing triangles, the local map of the posed mesh (the $J$ the mesh itself induces by deforming each triangle) is compared with the kernel's $J$. First term only: median error 4–6 %, p90 14–19 %, p99 37–50 %, concentrated in the wing's hand and the feather fans, where the weight transitions are short (small $L$) and the angles large. With the elastic term, p90 5–6.5 % and p99 11–13 %; what remains is the curvature of the blend within the triangle (LBS is not piecewise affine: $J$ changes within the triangle, and the linearisation at the centre does not capture it).

**relMSE** — relative mean squared error, $\frac1N\sum_i (x_i-y_i)^2/(y_i^2+10^{-2})$, averaged over channels (the definition in `shaders/athenea/reference/image_compare_hdr.slang`, the same as in the denoising literature); it penalises relative errors equally in dark and bright areas.

**The ablation as a methodological lesson** — negating, swapping, halving or doubling the gradients produces relMSE values ordered by the *magnitude* of the change (0.0887 halved < 0.0900 negated < 0.0907 swapped < 0.0983 doubled), not by whether the change is correct. When the dominant error is something else (the conversion: blurred feather cards, no relief), any perturbation is scored as "more difference". That is why correctness is shown by two tests that are sensitive to sign and axes: the analytic strip, and a consistency diagnostic between neighbours, in which the stored gradient predicts $\Delta w\approx\nabla w\cdot\Delta x$ with an RMS residual of 12.5 % (against 158 % with swapped axes and 100 % with no gradient, which is the trivial baseline: predicting zero change).

**The pixels** — p99 14–16 in 8-bit levels over about 10 % of the pixels: small differences localised in the wing's hand, visually identical at that distance; the gain is expected for gaussians that are large relative to their spacing (coarse LODs, close-ups of a bend).

*References:* code: `apps/athenea/src/CmdCompare.cpp` (`athenea compare`), `shaders/athenea/reference/image_compare_hdr.slang`; `docs/decisions.md` (section "The whole Jacobian of a skinned gaussian", paragraphs "On the sparrow" and "The 2-3 % is the metric's").
:::

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

::: {.nota}
**`ParticleField3DGaussianSplat`** — the OpenUSD schema for 3D Gaussian clouds (the `ParticleField` family of the volumes/particles domain, resolved in Hydra by `usdVolImaging`), with positions, orientations, scales, opacities and SH coefficients as attributes. athenea adds its own `primvars:athenea:splat:*` primvars (codeless schemas in `modules/usd/schemas`).

**From an own schema to UsdSkel** — at first the rig was stored with an applied schema of its own, `AtheneaSplatSkinningAPI`, because UsdSkel does not contemplate applying `SkelBindingAPI` to anything that is not a `UsdGeomPointBased`. The standard names are now used (`UsdSkelBindingAPI::Apply`, `CreateJointIndicesPrimvar(false, 4)`, etc.), so any UsdSkel reader recognises the prim as skinnable; the engine's own reader still accepts the old names.

**`skinningXforms` sampled in time** — one `matrix4f` array per instant, holding the matrices $X_j(t)$ from `UsdSkelSkeletonQuery::ComputeSkinningTransforms` (joint world transform times inverse bind, in skeleton space). It is the only animated datum; everything else (gaussians, influences, gradients) is constant, which enables the identity-based reuse of §5.6. `--range START:END[:STEP]` sets the instants (by default the stage's range, in steps of one time code).

*References:* [43] in §10; code: `modules/usd/src/Export.cpp` (blocks "THE BINDING IS UsdSkel'S" and "AND THE JOINTS' TRANSFORMS, WRITTEN DOWN"), `modules/usd/src/MeshStage.cpp` (`MeshStage::skeletonTransforms`); `docs/decisions.md` (sections "The file carries the rig, not the frames", "Step 3: a cloud is bound to its Skeleton the way the specification binds anything").
:::

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

::: {.nota}
**usdSkelImaging and ext computations** — in Hydra, UsdSkel is resolved with *scene indices*: `UsdSkelImagingPointsResolvingSceneIndex` adds under each mesh/curve/points prim two `HdExtComputation`s (an aggregator with the constant data and the computation with the animated data) whose output is the deformed `points`. A `ParticleField` is not `PointBased`, so it gets no computation; but the `Skeleton` is still resolved (a *resolved skeleton*) with its `skinningTransforms` at any instant.

**`HdAtheneaSkelSplatSceneIndex`** — a *filtering scene index* (`HdSingleInputFilteringSceneIndexBase`) inserted into the chain: for each cloud bound to a Skeleton, it injects into the prim the primvar `athenea:splat:skinningXforms` with the resolved transforms, remapped with `UsdSkelAnimMapper` if the cloud names a subset of joints, and propagates the Skeleton's *dirty* notices to its clouds (`_PrimsDirtied`). It is the "missing half" of usdSkelImaging for clouds.

**The cache and the priority** — the primvar written by the conversion is a cache: it lets a stage that composes the cloud without the Skeleton keep animating, and when both are present the cache wins (what is explicitly authored). The practical consequence is that a new Skeleton animation is not seen if the cloud carries a cache (the test in the next note checks this).

**Dual quaternions not implemented** — if the binding asks for `skel:skinningMethod = dualQuaternion`, `ParticleField.cpp` prints a warning once and uses LBS. For a cloud, DQS would also require deriving the Jacobian of the normalised dual-quaternion blend, considerably more complex than that of LBS.

*References:* [43] in §10; Kavan et al. (2008) for DQS; code: `modules/usd/src/SkelSplat.h`, `modules/usd/src/SkelSplat.cpp`, `modules/usd/src/RenderDelegate.cpp` (registration of the filter), `modules/usd/src/ParticleField.cpp` (the `dualQuaternion` warning); `docs/decisions.md` (section "Aligning with the specification, step by step", subsection "Step 3: a cloud is bound to its Skeleton the way the specification binds anything").
:::

**Measured.** 512 gaussians bound to a one-joint Skeleton whose animation
puts the joint at t = 1 where the cache has it at t = 2 draw, with the cache
blocked, the cache's t = 2 frame at t = 1 pixel for pixel (p99 0, max 0). On
the sparrow, the 609 joints the engine receives through the Skeleton agree
with the rig layer's; USD's own transforms against the Blender-written cache
differ by at most $5\times10^{-4}$, and the frames differ by a mean of
$8\times10^{-6}$.

::: {.nota}
**The priority test** — a case is built where Skeleton and cache disagree on purpose (the Skeleton puts the joint at $t=1$ where the cache puts it at $t=2$). With the cache locked (present and authored), frame $t=1$ must be pixel-identical to the cache's frame $t=2$: p99 0 and max 0 in `compareImages`. It shows that the "cache wins" rule holds deterministically.

**On the sparrow** — the 609 joints that arrive through the Skeleton match those of the rig layer, and the matrices computed by USD differ from those written from Blender by at most $5\times10^{-4}$ (the export's quantisation: translations and rotations to six decimals, scales as `half3`), which translates into a mean image difference of $8\times10^{-6}$.

*References:* code: `tests/usd/test_usd.cpp`; `docs/decisions.md` (section "Aligning with the specification, step by step", subsection "Step 3").
:::

**Time scale.** Without `timeCodesPerSecond` the cloud was read at 24 under a
30 fps stage and lagged the mesh (mean error 0.031 at frame 16, 0.049 at 61);
with the rate written, $1.4\times10^{-5}$ and $1.1\times10^{-5}$.

::: {.nota}
**`timeCodesPerSecond`** — layer metadata that says how many time codes there are per second. USD composes layers with different rates by rescaling their samples (an implicit *layer offset*): a layer without the metadata is read with the default value, 24. The cloud, sampled at 30 and composed under a stage at 30, was stretched by $30/24=1.25$ and fell behind cumulatively (hence the error growing from 0.031 at frame 16 to 0.049 at frame 61). Writing it (`MeshStage::timeCodesPerSecond()` → `SplatSkinning` → export) leaves the error at the quantisation level, $\sim10^{-5}$.

*References:* USD documentation, "Time Codes and Time Scaling" (`timeCodesPerSecond`, `framesPerSecond`); code: `modules/usd/src/MeshStage.cpp`, `modules/usd/src/Export.cpp`, `scripts/sparrow-anim.py`; `docs/decisions.md` (section "One cloud, seventy-one clips", subsection "And the cloud was playing slow").
:::

**Clips.** The sparrow arrives as five FBX files with 71 clips. Each clip
changes only `skinningXforms` (609 matrices a frame), so a clip is a layer
over one shared cloud: 377 MB for all 71, against 22 GB for 71 converted
clouds. `SparrowClips.usda` offers them as a USD **variant set**. The cloud
comes in as a payload, because a sublayer's local opinion would beat the
variant, and each clip comes in as a reference inside its variant. The engine
stays generic: it lists variant sets, selects one in an in-memory override layer, and
reports `animationRange()`, the span of authored samples (for example 1..25
for `land_hop_1`, 1..33 for `air_fly_A0`). Choosing a clip reads about 3 MB,
not 213 MB.

::: {.nota}
**A clip as a layer** — since a clip changes only `skinningXforms`, each clip is a layer with 609 matrices per frame. 71 clips = 377 MB (about 5.3 MB on average) against $71\times313$ MB $\approx22$ GB if the cloud were converted for each one.

**Variant set** — a USD composition mechanism: a prim carries a named set (e.g. `clip`) of variants; a *selection* decides which one is composed. The engine knows nothing about sparrows: it lists the variant sets, writes the selection into the session layer (the file does not change) and returns `animationRange()`, the actual interval of the written samples, instead of the root layer's `startTimeCode`/`endTimeCode` (which hold for the longest clip, 1..81, and would freeze a short clip).

**Payload versus sublayer, and LIVRPS** — USD's composition strength order is *Local, Inherits, VariantSets, References, Payloads, Specializes*. An opinion in a sublayer is *local* and would beat any variant: the cloud's default value of `skinningXforms` would mask each clip's samples. With the cloud brought in as a **payload** (the weakest except specializes) and each clip as a **reference inside its variant**, the clip's samples win. Moreover, USD opens only the selected variant's reference: choosing a clip reads about 3 MB, not the 213 MB of the cloud and everything else.

*References:* [43] in §10; USD documentation, "LIVRPS Strength Ordering" and "VariantSets"; code: `scripts/sparrow-clips.py`, `scripts/sparrow-clip-variants.py`, `modules/usd/src/StageRenderer.cpp` (`animationRange`, variant selection), `modules/mcp/src/Tools.cpp`; `docs/decisions.md` (sections "One cloud, seventy-one clips", "Seventy-one animations, and not one line about sparrows", subsections "Why the timeline stopped asking the stage" and "Packaging: a payload for the cloud, a reference a clip").
:::

| asset (`mesh2splat --skinned`) | gaussians | joints | instants | file |
|---|---|---|---|---|
| rigged tube, resolution 512 | 184 320 | 40 | 25 / 121 | 9.27 / 9.50 MB |
| Khronos fox, resolution 512 | 204 517 | 24 | 28 | 13.35 MB |
| sparrow, resolution 1400 (first conversion) | 4 991 908 | 609 | 33 | — |

::: {.nota}
**The table** — the tube and the fox (Khronos glTF-Sample-Assets' Fox, CC0/CC-BY) are mid-sized validation assets; the sparrow at resolution 1400 illustrates the real scale: almost 5 M gaussians and 609 joints. Its file size was not given because that first conversion was not kept as it was. "Resolution" is the number of cells of the conversion grid along the longest side of the box (§3). With two influences per vertex the tube's file would be 12.55 MB instead of 9.5: the barycentric blend gives most gaussians four joints instead of one, and joints + weights take 32 B per gaussian in float.

*References:* `docs/decisions.md` (sections "What a rigged cloud actually costs", "The fox").
:::

## 5.6 The engine: posing, bounds, refit and motion blur

`Engine::carryCloud` pairs the file's two arrays into the (joint, weight)
layout, keeps the uploaded cloud as the bind pose, and runs the skinner into a
separate `GpuSplats` that shares the harmonics and PBR buffers, since the
skinner writes neither. Measuring this exposed two errors. First, a posed
cloud's extent is not the bind pose's, so the extent is now folded again on
the device after skinning. Second, `Engine::bounds()` folded the uploaded
cloud instead of the drawn one.

::: {.nota}
**`Engine::carryCloud`** — it does three things: (1) it interleaves the file's two arrays (`jointIndices`, `jointWeights`) into (joint, weight) pairs, the format every skinner in the engine reads, and transposes the matrices; (2) it keeps the uploaded cloud as the bind pose and writes the posed one into buffers of its own, so that what a renderer sees does not change identity between frames; (3) it dispatches `scene::SplatSkinner` and bumps the cloud's `revision` so that structures built over the previous pose know they must update.

**The two bugs** — a posed cloud leaves its rest box (the raised wing sticks out), so everything that culls, frames or sorts by that box was looking at the wrong place; now the box is reduced again on the GPU after skinning. And `Engine::bounds()` folded `entry.gpu->bounds` (the uploaded cloud) instead of the drawn one.

*References:* code: `modules/usd/src/Engine.cpp` (`Engine::carryCloud`, `Engine::bounds`), `modules/usd/src/Engine.h`; `docs/decisions.md` (section "And the engine puts the cloud where the skeleton is").
:::

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

::: {.nota}
**Why every time code "was" a new cloud** — `usdVolImaging` marks the `ParticleField` as time-varying, so every step of the timeline reached `Sync` with `DirtyPoints | DirtyPrimvar`. `VtArray` is copy-on-write: a `Get` at another instant of an attribute without samples returns *the same buffer*; comparing the arrays' addresses and sizes (ten pointers plus the SH degree, `Engine::identityOf`) tells whether anything the decode depends on has changed. `skinningXforms` is excluded on purpose.

**`_held`** — in crate files (`.usdc`), compressed integer arrays are decompressed into a new buffer on every read, so the identity comparison would fail; `HdAtheneaParticleField` holds (`_held`) the first read of the arrays without time samples and reuses it.

**137 MB** — $4\,269\,858$ gaussians $\times$ 4 pairs $\times$ 8 B (two floats) $\approx136.6$ MB of pairs rebuilt on the CPU every frame; now only on upload.

**The table** — "median" includes frames with extra work (the first frame, recompilation); "warm" is the steady frame. Raster drops from 31.56 to 13.40 ms warm (about 75 fps with 4.27 M skinned gaussians). `rt` improves in the median but not warm: its remaining cost lay in the proxies and the acceleration structure, which is what refit addresses.

*References:* code: `modules/usd/src/ParticleField.cpp`, `modules/usd/src/ParticleField.h` (`_held`), `modules/usd/src/Engine.cpp` (`Engine::identityOf`, `CloudIdentity`); `docs/decisions.md` (section "A time change is not a new cloud").
:::

**Refit, not rebuild.** A pose changes neither the number of particles nor
their order, so `GaussianRayTracer::sync` refits. On the hardware route this
recomputes proxies and updates each chunk's BLAS in place. On the compute BVH
route it refits one tree height at a time, bottom up, with no read-back. A
full rebuild runs every `refitsPerRebuild` = 240 refits. Against a rebuild
every pose, the images agree to p99 0, max 1. A stale tree reads p99 255.
Sparrow at resolution 256 (300 862 gaussians, compute BVH): 140.9 / 143.1 /
135.6 ms per frame before, 84.7 / 83.9 / 83.7 ms after (−40 %).

::: {.nota}
**Refit on both routes** — *hardware*: the tracer builds triangle proxies for each gaussian (an enclosing icosahedron, in the style of 3DGRT [7]) and one BLAS (*bottom-level acceleration structure*) per chunk; when posing, it recomputes the proxies and updates the BLAS in place (`BuildMode::Update`, built with `AllowUpdate`; on Metal, `refitAccelerationStructure`). *Compute BVH*: Karras's LBVH [32] is refitted one height at a time, bottom-up (`bvh_refit_level.slang`), one dispatch per height. Since the number of heights is known at build time, the number of dispatches is fixed and no counter needs to be read back to the CPU (the original build iterated "until nothing changes", which required readbacks).

**Why rebuild every 240** — refit keeps the tree's topology: if the primitives move far from the build pose, the boxes grow and overlap, and tracing slows down. A periodic rebuild restores quality; 240 was measured as the point at which tracing has not yet degraded, and a rebuild costs a hitch of about 50 ms.

**The figures** — p99 0, max 1 against rebuilding every pose: one-level differences from walking the tree in another order (transmittance sums in float in another order). A tree *without* refit (the old pose's boxes) gives p99 255: the rays do not find the gaussians. Sparrow at 256 (300 862 gaussians): from ~140 ms to ~84 ms per frame, −40 %.

*References:* [7], [32] in §10; Wald, Boulos, Shirley (2007), "Ray Tracing Deformable Scenes Using Dynamic Bounding Volume Hierarchies", ACM TOG; code: `modules/render/include/athenea/render/GaussianRayTracer.h` (`refitsPerRebuild = 240`), `modules/render/src/GaussianRayTracer.cpp` (`GaussianRayTracer::sync`), `shaders/athenea/rt/bvh_refit_level.slang`, `tests/render/test_splat_render.cpp`; `docs/decisions.md` (section "A posed cloud refits its ray tracing structure").
:::

**Motion blur from the skeleton.** `ParticleField` samples
`skinningXforms` over the shutter. The skinner blends the second pose in the
same loop (points only, about a third of a kernel) and writes each gaussian's
displacement as two words of halves. The rasteriser turns that displacement
into a rank-one covariance term. Hydra brackets rather than clips the
shutter, so the displacement is scaled by $(t_c-t_o)/(t_1-t_0)$; without that
factor the wings blur twice as far. On the 5.9 M sparrow at 1280×800, a
frame takes 90.0 ms with shutter 0 and 100.5 ms with shutter 0.5 (+11 %).

::: {.nota}
**A gaussian's motion blur as a rank-one term** — the image of a gaussian dragged uniformly along a segment $d$ (on screen) during the shutter is its convolution with that segment. Matching second moments, a uniform segment of length $|d|$ has variance $|d|^2/12$ along itself, so $\Sigma'_{2D}=\Sigma_{2D}+dd^{\mathsf T}/12$: a rank-one update of the 2D covariance. Convolution with a normalised measure preserves the integral, so the peak must drop by $\sqrt{\det\Sigma/\det\Sigma'}$, the same root the Mip-Splatting filter [4] already applies. Discarded alternative: averaging several sub-samples of the mean (*Gaussian Splatting on the Move*, ECCV 2024, uses five).

**The displacement from the skeleton** — `ParticleField` samples `skinningXforms` at the two samples that span the shutter. The `splatSkin` kernel reuses the joint loop to blend the second pose (only the point, not the frame: `movedEnd`), which costs "a third of a kernel". The displacement is stored in cloud space as two words of halves (`motion`).

**The scale factor** — Hydra delivers the samples $t_0,t_1$ that *span* the shutter interval $[t_o,t_c]$, not clipped to it. The measured displacement corresponds to $t_1-t_0$, whereas the blur must correspond to $t_c-t_o$: it is multiplied by $(t_c-t_o)/(t_1-t_0)$. With a shutter of 0.5 between frames, the displacement is doubled without the factor.

**Cost** — +11 % (90.0 → 100.5 ms) for the second pose and the term in the projection.

*References:* [3], [4] in §10; Seiskari et al. (2024), "Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation", ECCV; code: `shaders/athenea/scene/splat_skin.slang` (`skinningXformsEnd`, `motion`), `shaders/athenea/splat/splat_project.slang` (`d.x * d.x * (1.0 / 12.0)`), `modules/usd/src/ParticleField.cpp` (samples spanning the shutter); `docs/decisions.md` (section "A splat under a shutter: one rank-one term, not five passes").
:::

## 5.7 Light that turns with a limb: zonal transfer lobes

A static cloud can carry a precomputed transfer of nine spherical harmonics
in world space. Under a skeleton those harmonics would keep the sky of the
conversion pose, and rotating them costs a $9\times9$ matrix per gaussian.
athenea instead stores one or two **zonal lobes** per gaussian, with each
lobe's axis expressed in the gaussian's own frame. A zonal function rotates by
rotating its axis, a single $3\times3$, so whatever frame the skinner
produces carries the lobes with it, and no gaussian has to be assigned to a
part.

::: {.nota}
**Precomputed transfer (PRT)** — the function $T(\omega)=V(\omega)\max(0,n\cdot\omega)$ (visibility times cosine) projected onto the spherical harmonics basis; with the environment light projected as well, irradiance is a dot product of coefficients [14]. With three bands ($l=0,1,2$) there are nine coefficients, enough for low-frequency diffuse lighting [17].

**Rotating SH** — rotating a function expressed in SH requires one matrix per band ($1\times1$, $3\times3$ and $5\times5$ blocks: a block-diagonal $9\times9$), whose entries are the Wigner-D matrices; computing them per gaussian per frame is expensive (Ivanic and Ruedenberg, 1996, give a recurrence). Alternative: rotate per *part* (one matrix per bone), but then a gaussian on the boundary between two parts does not know which one to use.

**Zonal harmonics (ZH) and zonal lobes** — a function with rotational symmetry about an axis $a$ has, in the system where $a$ is the pole, only $m=0$ components: three numbers $z_0,z_1,z_2$ for three bands. Rotating it means rotating its axis: one $3\times3$ applied to $a$. So, if the axis is stored in the gaussian's *own frame*, the frame the skinner produces (§5.4.2) carries it automatically. A real transfer is not symmetric, so it is approximated by the sum of one or two zonal lobes (as in Sloan [16]). The same idea appears in *Relightable Full-Body Gaussian Codec Avatars* (arXiv 2501.14726), which stores transfer in ZH for the same reason.

**No part assignment** — unlike per-part rotation, here each gaussian carries its own frame, so there is no boundary between parts.

*References:* [14], [16], [17] in §10; Ivanic, Ruedenberg (1996), "Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion", J. Phys. Chem.; Green (2003), "Spherical Harmonic Lighting: The Gritty Details", GDC; code: `plugins/splattransferzonal/splattransferzonal.slang` (header), `shaders/athenea/splat/splat_relight.slang` (`splatTransferFrame`); `docs/decisions.md` (section "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame").
:::

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

::: {.nota}
**The lobe in SH** — a zonal function with coefficients $z_l$ and axis $a$ has, in the SH basis, coefficients $g_{lm}=z_l\,k_l\,Y_{lm}(a)$ with $k_l=\sqrt{4\pi/(2l+1)}$. This follows from the **addition theorem** for spherical harmonics: $P_l(a\cdot\omega)=\frac{4\pi}{2l+1}\sum_m Y_{lm}(a)Y_{lm}(\omega)$, with $P_l$ the Legendre polynomial. The factor $k_l$ normalises so that $\sum_m g_{lm}^2=z_l^2$ (the band's energy is $z_l^2$ regardless of the axis).

**Best coefficients for an axis, and the residual** — projecting $f$ onto the subspace of zonal functions with axis $a$ means taking $z_l=k_l\sum_m f_{lm}Y_{lm}(a)$ (the dot product of $f$ with that band's unit lobe). Since the projection is orthogonal, $\|f-g\|^2=\|f\|^2-\|g\|^2=\|f\|^2-\sum_l z_l^2$: minimising the error means maximising the captured energy $E(a)=\sum_l z_l^2$. $E(a)=E(-a)$ (flipping the axis flips the sign of $z_1$ but not its square), so searching one hemisphere is enough.

**The search** — non-linear and with local optima, so it is seeded with reasonable candidates: the band-1 direction (the vector $(f_{1,1},f_{1,-1},f_{1,0})$ in $x,y,z$ order, the transfer's "dominant direction"), the gaussian's normal and 64 directions of a Fibonacci spiral (nearly uniform points over the hemisphere); then a local walk that halves its step when it does not improve. With two lobes: the first is fitted, the second is fitted to the residual $f-g_1$, and they are refitted alternately twice (block coordinate descent, Sloan's "fast ZH approximation" [16]).

**Storage** — 10 floats: per lobe, the axis in octahedral coordinates $(u,v)$ (the unfolded octahedron mapped to a square: two numbers for a unit vector, with no pole singularities) and $z_0,z_1,z_2$; $2\times(2+3)=10$. On the device it goes in the usual `transfer` buffer as f16 pairs, and it is `transferCount` = 10 that tells it apart from nine SH (9) or from the two halves (36): no new flag. `splatTransferFrame` decodes each axis, takes it to the world with the gaussian's current rotation and rebuilds the nine coefficients with the formula above, so the rest of the shading (`transferValue`) does not change.

**Fit error** — the AOFX effect attaches a histogram of $\|f-g\|/\|f\|$ in quarter octaves, which the conversion prints as median, p90 and p99.

*References:* [16] in §10; Cigolle et al. (2014), "A Survey of Efficient Representations for Independent Unit Vectors", JCGT (octahedral encoding); González (2010), "Measurement of Areas on a Sphere Using Fibonacci and Latitude–Longitude Lattices", Math. Geosci.; code: `plugins/splattransferzonal/splattransferzonal.slang` (`zonalAbout`, `kept`, `takeAway`, `bestAxis`, `tzFit`), `plugins/splattransferzonal/SplatTransferZonal.cpp`, `shaders/athenea/splat/splat_relight.slang` (`splatTransferFrame`), `shaders/athenea/usd/transfer_zonal_io.slang`; `docs/decisions.md` (section "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame", paragraphs "What is stored" and "The fit").
:::

**The bake pose.** At first the bake posed the cloud at `--time` using four
influences per gaussian, while Hydra skinned the mesh with all of its own
(ten per vertex on the feathers, 25 on the wings). Where the two disagreed,
the bake's ray missed its surface, and 566 603 of 1 504 160 gaussians came
out transparent. The bake now holds the skeleton in its bind pose
(`StageRenderer::holdBindPose`) and poses the cloud with identity transforms:
1 504 156 of 1 504 160 find their surface. The transfer therefore holds the
bind pose's self-shadowing (wings spread). Occlusion that another limb casts
in another pose is the job of §5.8.

::: {.nota}
**Why the bake failed in an animated pose** — the bake casts rays from each gaussian against the posed *mesh* (Hydra deforms it with all of its influences: 10 per vertex on the feathers, 25 on the wings), while the gaussian had been posed with only four. Where the two positions disagree by more than the ray's tolerance, the ray does not find its surface and the transfer is left empty; an empty transfer is written as transparent: 566 603 of 1 504 160 gaussians (37.7 %) with opacity 0, plumage seen as "loose dots".

**The fix** — in the bind pose there is no deformation, and mesh and cloud coincide exactly: `StageRenderer::holdBindPose` writes into the session layer the skeletons without animation (rest transforms = bind transforms made local), and the cloud is posed with identity transforms. Result: 1 504 156 of 1 504 160 (99.9997 %). Since the lobes live in the gaussian's frame, a transfer baked in the bind pose turns with the bone just like one baked in any other pose; what changes is *which* self-shadowing it contains: that of the extended wings.

*References:* code: `modules/usd/include/athenea/usd/StageRenderer.h`, `modules/usd/src/StageRenderer.cpp` (`holdBindPose`), `apps/athenea/src/CmdMesh2Splat.cpp`; `docs/decisions.md` (section "A skinned transfer is traced at the bind pose").
:::

**Checked.** In `[zonal]`, a two-lobe transfer survives the fit and f16
storage within 0.5 %. Turning a gaussian's frame by $R$ and reading along
$Rw$ gives what the unturned gaussian gives along $w$. In a USD test, 512
gaussians carried by a joint that turns 60° match the same cloud under an
Xform of that rotation, with relMSE below $10^{-4}$ on both raster and traced
routes. The fit's error on the sparrow has not been measured. The indirect
half of the transfer is not kept zonal.

::: {.nota}
**The checks** — (1) *round trip*: a transfer that is already a sum of two lobes must be recovered after the fit and the f16 quantisation within 0.5 %; (2) *equivariance*: turning the frame by $R$ and reading along $Rw$ must give the same as reading the unturned one along $w$, i.e. $T_{R\cdot}(Rw)=T(w)$, which is the property that justifies the design; (3) *end to end in USD*: 512 gaussians carried by a joint that turns $60^\circ$ about $(1,2,0.5)$ against the same cloud standing still under an `Xform` with that rotation, under a sky that differs in each direction, raster and traced: relMSE $<10^{-4}$.

**The indirect half** — athenea's transfer separates direct and indirect light (the "two halves", §4); the indirect one is not stored in zonal form, so a skinned cloud has only direct transfer, which explains why the body comes out somewhat darker in §5.10.

*References:* code: `tests/aofx/test_transfer_zonal.cpp` (case "a zonal transfer is fitted in the gaussian's frame and turns with it"), `shaders/athenea/test/transfer_zonal_check.slang`, `tests/usd/test_usd.cpp` (tag `[zonal]`); `docs/decisions.md` (section "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame", subsection "How it is checked").
:::

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

::: {.nota}
**Why the obvious routes fail** — on the traced route, a splat's shading had no shadow pass of its own (`rt_shade.slang` declared the parameter without using it), so `--splat-shadows` changed nothing. On raster, the shadow pass traces from each splat towards the light and starts the ray at an offset of `shadowOffset` of the splat's own sigmas. That value (3 sigmas) is meant for photogrammetric captures of fat, sparse splats; in a converted cloud the splats measure $\sim1.6\times10^{-4}$, and the plumage stacks many layers within half a millimetre: the transmittance, a product $\prod(1-\alpha_i)$, falls almost to zero after crossing a few neighbours (with $\alpha\approx0.91$ per layer, three layers leave $0.09^3\approx7\times10^{-4}$). The sweep from 1 to 800 sigmas with no plateau shows that no correct offset exists: any value is at once too short for some gaussians and too long for others.

**The literature** — baked visibility survives articulated motion only if it is factored by body part: Lin et al. [41] train a directional visibility MLP per part (*part-wise light visibility*, in relightable neural avatars); DNF-Avatar [42] uses a scalar ambient-occlusion grid per bone. athenea combines both: *directional* tables (as in [41]) in explicit per-part *grids* (as in [42]), with no neural network.

*References:* [41], [42] in §10 (probably Lin et al., "Relightable and Animatable Neural Avatars from Videos", AAAI 2024, and Jiang et al., "DNF-Avatar: Distilling Neural Fields for Real-time Animatable Avatar Relighting", 2025; the paper cites them without full details); code: `shaders/athenea/rt/rt_shade.slang`, `shaders/athenea/splat/splat_project.slang`; `docs/decisions.md` (sections "A cloud does not shadow itself, and the offset cannot be tuned to fix it" and "A cloud shadows itself by part: baked once, read every frame, no ray").
:::

**Partition.** `partitionJoints` takes the largest subtree not yet a part,
with at least `minJoints` (6) joints, and makes it a part, until there are
`--parts` (12). On the sparrow this gives root, centre, chest, pelvis, head,
and each wing in three segments. Each gaussian belongs to the part of its
dominant joint.

::: {.nota}
**`partitionJoints`** — the skeleton is a tree (the hierarchy is deduced from the joint paths: the parent of `A/B/C` is `A/B`). The size of each subtree is computed; the largest root is taken as a part, and then, repeatedly, the largest subtree that is not yet a part head and has at least `minJoints` (6) joints, until there are `--parts` (12). Each joint belongs to the part of its nearest head ancestor. Choosing the largest first separates a wing from the chest before a toe from the foot: large partitions are the ones that cast large shadows. On the sparrow they come out as: root, centre, chest, pelvis, head, and each wing in three segments (two parts, root and centre, carry no gaussian).

**Dominant joint** — each gaussian is assigned to the part of its heaviest joint (`visibilityPartOf`). A gaussian on a boundary between parts ends up in just one, an acceptable approximation because the parts are large.

*References:* code: `modules/technique/src/SplatVisibility.cpp` (`partitionJoints`), `modules/technique/include/athenea/technique/SplatVisibility.h`, `apps/athenea/src/CmdVisibility.cpp` (`--parts`, `--min-joints`, defaults 12 and 6), `tests/technique/test_splat_visibility.cpp` (case "a rig becomes parts by its largest subtrees, small ones staying with their parent"); `docs/decisions.md` (section "A cloud shadows itself by part: baked once, read every frame, no ray", subsection "What it is").
:::

**Bake** (`athenea visibility`, `splat_visibility_bake.slang`). This is done
once, in the bind pose. For part $p$, a grid of $N^3$ probes covers the
cloud's box plus a 5 % margin. Each probe holds an octahedral map of $O^2$
directions. A texel stores the transmittance of a ray leaving the probe
through part $p$'s gaussians alone. The part's gaussians are gathered into a
cloud of their own and traced with inline rays, using the same product of
$(1-\alpha)$ that the shadow kernels use. Each probe also stores an ambient
term, its mean over directions, used for domes.

::: {.nota}
**Per-part visibility field** — a function $F_p(x,\omega)\in[0,1]$: the transmittance from point $x$ in direction $\omega$ through part $p$ *only*, in the bind pose. It is discretised as an $N^3$ grid of probes (like *irradiance probes* or the *light field probes* of McGuire et al., 2017) and, at each probe, an octahedral map of $O\times O$ directions: the unfolded octahedron covers the whole sphere with a square, without extreme distortion or pole seams.

**The bake** — `visBake`: one thread per (probe, direction), $N^3O^2$ threads per part. The part is gathered into a cloud of its own over which the tracer builds proxies (icosahedra) and a BLAS; each thread launches an inline `RayQuery` (no hit shaders) and accumulates transmittance with `shadowAlpha`, the same function the shadow kernels use: the gaussian's maximum $\alpha$ along the ray, and $T=\prod(1-\alpha_i)$, stopping when $T$ falls below a threshold. The ray starts one probe cell away (`start`).

**Ambient term** — the mean of the probe's transmittance over all its directions, a single number approximating the fraction of sky not covered by the part: it is used for dome lights.

*References:* McGuire et al. (2017), "Real-Time Global Illumination using Precomputed Light Field Probes", I3D; Cigolle et al. (2014) for the octahedral map; [7] in §10 (Gaussian ray tracing); code: `shaders/athenea/splat/splat_visibility_bake.slang` (`visBake`), `shaders/athenea/splat/splat_visibility_common.slang`, `modules/technique/src/SplatVisibility.cpp` (`SplatVisibility::bake`), `apps/athenea/src/CmdVisibility.cpp`; `docs/decisions.md` (section "A cloud shadows itself by part: baked once, read every frame, no ray").
:::

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

::: {.nota}
**The inverse map per part** — the posed position is $x'=X_k\,G\,x_{\text{rest}}$ (with the part treated as rigid, governed by its head joint $k_p$). Inverting: $Gx_{\text{rest}}=A_k^{-1}(x'-t_k)$ and, with $G=(L(G),g)$, $x_{\text{rest}}=L(G)^{-1}\big(A_k^{-1}(x'-t_k)-g\big)=UA_k^{-1}(x'-t_k)-Ug$. It is an affine transform $T_p^{-1}$, identical for all the gaussians in a frame.

**Why precompute** — the old kernel inverted two $3\times3$ matrices (the joint's and the bind's) per gaussian, per part and per light: $\approx5.9\times10^6\times11\times2\times2\approx2.6\times10^8$ inversions per frame. `visPartFrames` computes them once per part (12 threads) and writes three `float4` rows per part into `partFrames`; `visFactors` only applies a $3\times4$. The read dropped from 12.7 to 8.0 ms.

**The product of visibilities** — the parts are assumed to occlude independently, so the total transmittance towards the light is the product of each part's: $V=\prod_p F_p$. It is exact if the ray segments crossing each part do not overlap in depth with others (transmittance is multiplicative along the ray: $T_{AB}=T_AT_B$); it is approximate when two parts interpenetrate. Each $F_p$ is read in *that* part's bind space: position through $T_p^{-1}$ and direction through the inverse linear part (a direction is not translated).

**Interpolation** — trilinear between the 8 probes surrounding the point and, at each, bilinear between 4 texels of the octahedral map: 32 reads per part and light (`fieldAt`). Outside the field's box it is 1 (no occlusion). For a non-dome light *one* direction per light is taken (the central sample of `sampleLight`), so an area light casts a shadow from a single direction; a dome uses the ambient term.

**`shadowFactors`** — the per-(gaussian, light) factor buffer that the raster route (`splat_project.slang`) and the traced route (`rt_shade.slang`) already read: the traced route gets splat-on-splat shadowing for the first time. Zero rays per frame: everything is table reads.

*References:* Max (1995), "Optical Models for Direct Volume Rendering", IEEE TVCG (multiplicative transmittance); code: `shaders/athenea/splat/splat_visibility_read.slang` (`visPartFrames`, `partFrames`, `inverse3`, `fieldAt`, `ambientAt`, `visFactors`), `modules/technique/src/SplatVisibility.cpp` (`SplatVisibility::factors`), `shaders/athenea/splat/splat_relight.slang`, `shaders/athenea/rt/rt_common.slang`, `shaders/athenea/rt/rt_shade.slang`; `docs/decisions.md` (section "A cloud shadows itself by part: baked once, read every frame, no ray").
:::

**Found while building it.** An empty part's unwritten texels read as total
shadow and turned the bird black; a grid of 0 now marks it and it is skipped.
The bake traces inline rays, which the CUDA target lacks, so it is a separate
module and CUDA devices read fields baked elsewhere. A part does not shadow
itself: its field, read at its own gaussian, is the crowd it stands in. An
$8\times8$ map (about 22° per texel) cannot resolve a wing's edge; 16 and 32
are exact outside a band of one probe cell.

::: {.nota}
**Empty part** — if a part has no gaussians, its bake writes nothing and its texels stay at 0, which as a transmittance means "black from every direction". A grid of size 0 marks the part, and the read skips it (`field.grid == 0u`).

**CUDA and inline rays** — `RayQuery`/`TraceRayInline` does not exist in Slang's CUDA target; if the bake were in the same module as the read, the read would not compile either. Kept apart, a CUDA device (the L4) reads fields baked on Metal: the fields travel in the file (`primvars:athenea:splat:visibilityParts`, `visibilityTexels`, `visibilityAmbient`, `visibilityPartOf`). Metal also had a limit of 31 buffers per pipeline, which led to splitting off the read module.

**A part does not shadow itself** — reading the part's own field at one of its gaussians measures the "crowd" it is immersed in: the same problem that turned the ray-traced shadow black, one level up. Local self-shadowing (feather on feather within a wing segment) is lost.

**Angular resolution** — an $8\times8$ map has 64 directions for $4\pi$ sr: $\approx0.2$ sr per texel, about $180^\circ/8\approx22^\circ$ a side. The penumbra it gives at the wing's height is wider than the wing. With 16 and 32 the answer is exact except in a band of one probe cell (plus the wing's height times the tangent of the light's inclination).

*References:* Slang documentation (capabilities per target: `RayQuery` unavailable on CUDA); code: `shaders/athenea/splat/splat_visibility_bake.slang` (header comment), `shaders/athenea/splat/splat_visibility_read.slang`, `tests/technique/test_splat_visibility.cpp` (case "a wing's field shadows the body under it, and moves with the wing's joint"); `docs/decisions.md` (section "A cloud shadows itself by part…", subsection "What was found building it").
:::

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

::: {.nota}
**The figures** — $24^3\times16^2=13\,824\times256=3\,538\,944$ texels per part; with 12 parts, $42\,467\,328$ texels; in f16 (2 B), $84.9\times10^6$ B $=81.0$ MiB: the "81 MB" are MiB. A 22 s bake on the M5 Pro for the 12 fields. The 21.3 % of factors below ½ indicates how much surface is noticeably shadowed at frame 31 with sun and dome.

**Cost per frame** — warm, raster actually *drops* (13.40 → 9.69 ms), a sign that the read's cost lies below the run-to-run variation; in the median it rises (56.87 → 71.82 ms). On the L4 the read costs 5–7 ms per frame (73.04 against 67.48 ms on raster).

*References:* `docs/decisions.md` (section "A cloud shadows itself by part…", subsection "What was measured").
:::

On the L4 the fields cost 5–7 ms per frame for
4.27 M gaussians × 2 lights × 11 parts of lookups. The shadow lands where the
mesh's does (the far wing, the flank under the raised wing, the tail root) and
follows the wings across frames. With every feather card included (9.5 M
gaussians at resolution 1400), the bake ran out of memory at the largest part
on both the M5 Pro (Metal) and an L4 through Vulkan (a 1.17 GB BLAS for about
3.3 M gaussians). The bird is therefore baked at resolution 1100.

::: {.nota}
**Scale of the queries** — $4.27\times10^6$ gaussians $\times$ 2 lights $\times$ 11 parts (all but the gaussian's own) $\approx9.4\times10^7$ field reads per frame, each of 32 texels: 5–7 ms on an L4. The resulting shadow falls in the same place as the mesh's (far wing, flank under the raised wing, root of the tail).

**The bake's memory limit** — the bake of a part builds a BLAS over its gaussians; with all the feather cards (9.5 M gaussians at resolution 1400) the largest part has about 3.3 M gaussians and the BLAS takes 1.17 GB (with icosahedral proxies: $3.3\times10^6\times20$ triangles, about 18 B per triangle), which exhausts memory on both Metal and Vulkan. That is why the bird is baked at resolution 1100. Baking by chunks of a part (several BLASes) is what is missing (§5.13).

*References:* `docs/decisions.md` (section "A cloud shadows itself by part…"; section "A mesh past the ceiling is converted in slices, and a barb keeps its alpha").
:::

**Limits.** A part's own local shadowing (feather on feather) is left out.
The ambient term is a mean over the sphere rather than a cosine lobe. Parts
are assumed rigid and to occlude independently. The floor under the bird is
shadowed by a separate light-space map, which is stabilised against the
"breathing" of a posed bounding box (sized from the rest sphere, snapped to
the world texel grid). A density-grid alternative is not implemented.

::: {.nota}
**The four approximations** — (1) no local self-shadowing within a part; (2) the ambient term as a uniform mean over the sphere, not weighted by the cosine with the receiver's normal (the correct quantity for diffuse irradiance would be $\frac1\pi\int V(\omega)\max(0,n\cdot\omega)\,d\omega$); (3) rigid parts: LBS deformation within a part is not taken into account; (4) independence of occlusion (the product). The ground shadow is another mechanism: a shadow map in light space whose size is set from the rest sphere and snapped to the world's texel grid, so that it does not "breathe" with the posed box (which changes with every wingbeat and would make the shadow shimmer).

*References:* `docs/decisions.md` (sections "A cloud shadows itself by part…", subsection "What it is not"; "A cloud's shadow that does not breathe with the wings"; "A cloud's shadow, measured from the light, with no ray and no sort").
:::

## 5.9 Levels of detail and decimation for skinned clouds

The `.athc` octree merges Morton-ordered cells. That cannot apply to a skinned
cloud, because a cell that took wing and body together would not know which
joint to follow. Two mechanisms replace it.

::: {.nota}
**Morton-ordered octree** — the `.athc` LOD (§6) sorts the gaussians by Morton code (interleaving the bits of the quantised coordinates, which turns 3D proximity into 1D proximity), and each octree cell is a contiguous run of that order; merging a cell means replacing its gaussians with one that preserves moments. The order is computed on fixed positions. In a skinned cloud, a cell could mix wing and body gaussians: the resulting gaussian would have no coherent rig (does it follow the wing or the body?) and would tear when animated.

*References:* [2], [8], [32], [34] in §10; Morton (1966), "A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing"; code: `modules/lod/include/athenea/lod/Lod.h`, `modules/lod/src/Lod.cpp`; `docs/decisions.md` (section "The sun comes out of the sky and is lit as a light", subsection "Levels of detail for a cloud that moves").
:::

**Discrete levels by reconversion.** `mesh2splat --lod-levels N` converts at
the requested resolution, then at half of it, and so on, each level a cloud
with its own rig. `Engine::lodLevelsFor` draws the coarsest level whose cell
spans no more than `athenea:lod:threshold` pixels (default 1) at the nearest
point of its *posed* bounds. Undrawn levels are not posed (`deferredPose`).
Sparrow clips at 1100 / 550 / 275 cells (5.9 M / 1.5 M / 0.4 M gaussians),
1600×1000: 49.6 → 51.5 ms when the bird fills the frame; **71.3 → 24.7 ms** at
four times the distance. Every level stays resident with its own visibility
fields (81 MB each), and switching between levels is a step, not a blend.

::: {.nota}
**Discrete LOD by reconversion** — each level is an independent conversion at half the resolution, with its own rig (influences recomputed from the mesh, not averaged). `usd::writeLodAssembly` writes a stage that references the levels (`<name>_lod<n>.usdc`) with `athenea:lod:group` and `athenea:lod:cell` (cell size).

**Selection** — `Engine::lodLevelsFor` projects each level's cell at the nearest point of the *posed* bounds and draws the coarsest one whose cell covers at most `athenea:lod:threshold` pixels (1 by default): a screen-space error criterion, as in the classic LODs of Nanite [35] or of hierarchical splats [2]. Levels that are not drawn are not posed (`deferredPose`: the pending pose is kept and applied when a view asks for it), so they cost no skinning kernel.

**The figures** — when the bird fills the frame, the level chosen is the finest (its cell already covers about 2 px) and the time barely changes (49.6 → 51.5 ms); at four times the distance level 1 (1.5 M) is chosen and the frame drops from 71.3 to 24.7 ms. The cost: memory (all levels resident, each with its 81 MB visibility fields), and a jump, not a fade, when the level changes.

*References:* [2], [35] in §10; code: `modules/usd/src/Engine.cpp` (`Engine::lodLevelsFor`, `deferredPose`), `modules/usd/include/athenea/usd/Export.h` (`writeLodAssembly`); `docs/decisions.md` (subsection "Levels of detail for a cloud that moves").
:::

**Decimation that keeps the rig.** `usd::decimateStage` finds every
per-gaussian array and merges each by kind (`lod_attributes.slang`). When
`skel:jointIndices` has a matching `skel:jointWeights`, the pair is merged as
a rig (mode 3). Over the run of records $R$ that a kept gaussian stands for,
each with merge weight $\omega_r$,

$$c_j=\sum_{r\in R}\omega_r\sum_k w_{rk}\,[\,j_{rk}=j\,],$$

collecting up to 16 distinct joints. The `width` heaviest are kept and
renormalised, $w'_j = \min(c_j/\sum_{\text{kept}} c,\,1)$: a division on the
device is not correctly rounded, and a merge wholly on one joint came out at
$1 + 10^{-7}$. A record's dominant joint is
also folded into the merge key, so a merge never crosses a joint's edge.
Checked: a floor whose halves belong to two joints goes from 4096 to 256
gaussians, each kept gaussian wholly on its side's joint. The sparrow's finest
level (5 887 323 gaussians, with ids, material, visibility parts and rig)
keeps 87.7 % in 28 s. Posed at time 20, it draws as the undecimated cloud at
an image RMS of 0.002.

::: {.nota}
**`usd::decimateStage`** — reduces a cloud by merging gaussians (§6) and, for each per-gaussian array in the file, chooses how to combine it (`lod::AttributeMerge`): weighted mean (mode 0), "the first" for what cannot be blended (ids, parts; mode 1), bits by majority (2), rig (3), renormalised direction (4), zonal lobes (5). The kernel is `lodMergeAttribute` in `lod_attributes.slang`.

**Rig mode** — for each kept gaussian, which represents a run $R$ of original records with merge weights $\omega_r$ (the same ones that weight its moments), the weight each joint carries is accumulated: $c_j=\sum_r\omega_r\sum_kw_{rk}[j_{rk}=j]$ (the Iverson bracket is 1 if the record's $k$-th joint is $j$). Up to 16 distinct joints are collected (`kMostJoints`), the `width` (4) heaviest are kept and renormalised.

**The $\min(\cdot,1)$** — division on the GPU is not correctly rounded (Metal and CUDA use reciprocal approximations with up to a couple of ulp of error), and $c/c$ could give $1+10^{-7}$; the clamp avoids weights slightly above one, which would break test invariants and strict readers.

**The dominant joint in the key** — the merge key (what decides which records may be fused) includes the dominant joint, so a merge never crosses the boundary between joints: the wing+body cell problem of §5.9 is ruled out by construction. Test: a ground of two halves with different joints, 4096 → 256 gaussians, each one whole on its side's joint.

**The sparrow** — 5 887 323 gaussians with ids, material, visibility parts and rig are reduced to 87.7 % in 28 s; posed at $t=20$, an image RMS of 0.002 against the undecimated cloud.

*References:* [9], [34] in §10; code: `modules/usd/src/Decimate.cpp` (plans `lod::AttributeMerge::Rig`), `modules/usd/include/athenea/usd/Export.h` (`decimateStage`), `shaders/athenea/lod/lod_attributes.slang` (`lodMergeAttribute`, `kMostJoints`), `tests/usd/test_usd.cpp`; `docs/decisions.md` (sections "Decimation: the levels of detail, cut once by what a merge loses" and "A decimation keeps everything a cloud carries").
:::

## 5.10 The sparrow

The sparrow and a BMW were the first two assets converted that were not
authored for this renderer, and most of what it uncovered was outside the skinning maths (Figure 15).

::: {.nota}
**Assets "not made for this renderer"** — until the sparrow and a BMW, the test assets were built by hand or came from Pixar (Kitchen Set, etc.) and followed the engine's conventions. An imported production asset (here, from FBX to Blender to USD) brings foreign conventions: distinct rest and bind poses, premultiplied normal maps, alpha on a second UV set, wrongly assigned metallic. Most of the defects found have nothing to do with the skinning mathematics but with the export and reading chain.

*References:* `docs/decisions.md` (sections "Two assets out of Blender, and the two bugs they found", "What the sparrow needed on Blender's side").
:::

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

::: {.nota}
**Rest versus bind** — in UsdSkel, `restTransforms` (local) describe the rest pose and `bindTransforms` (world) the pose in which the mesh was bound; Blender writes the former from the armature's rest pose and the latter from the bind pose. UsdSkel computes the skinning matrix as $W(t)\,B^{-1}$. The script resolved a change of basis against *rest*, which cancels against rest but not against bind: the joints where the two differ (toes, head, beak; worst discrepancy 1.989 at `Toe.L.010`, close to half a turn) were left with a spurious rotation. The previous version's "round-trip check" did not catch it because the basis was *defined* so that the rest round trip worked: a tautological test.

**The correct formula** — with $P(t)$ the bone's evaluated world matrix in Blender (`pose.matrix`), $L$ its rest matrix in armature space (`matrix_local`) and $B$ the `bindTransform`, writing $W(t)=P(t)L^{-1}B$ makes $W(t)B^{-1}=P(t)L^{-1}$, exactly Blender's skinning matrix, without depending on either side's bone-axis convention. The residual error ($4.9$–$7.1\times10^{-5}$) is the file's quantisation (six decimals, scales as `half3`). In addition, USD's joint names are the bone names with characters that are invalid in identifiers replaced by `_` (e.g. `Spine.001_Pelvis` → `Spine_001_Pelvis`).

**Corrected stage** — `SparrowBird.usda` (not `SparrowFlight.usda`) applies the material fixes: feathers and eyes without metallic, an "unpremultiplied" normal map (`UV2_treesparrow_normal_fixed.png`, with the RGB channels not multiplied by alpha, which had biased the normals at edges) and the feathers' opacity read through the second UV set (`UVMap_001`).

**Cut-outs and budget** — honouring the `opacity` connected to the alpha (the feather cards are rectangles with cut-out alpha) avoids seeding gaussians in the transparent areas: from 15.4 M to 5.9 M at resolution 1400. Meshes that would exceed the ceiling of 2 M gaussians per conversion pass are processed in slices.

**Mesh reference** — to compare the cloud with the mesh one must first make sure that raster and path trace of the *mesh* agree: making them agree lowered the relMSE between the two from 7.28 to 3.67.

*References:* code: `scripts/sparrow-anim.py`; `docs/decisions.md` (sections "The sparrow's animation, and a skinning matrix that has to be true", "A feather is a hole in a rectangle", "The feathers read their shape by a second set of coordinates", "A mesh past the ceiling is converted in slices, and a barb keeps its alpha", "The sparrow's mesh, the same bird on both routes").
:::

Measured cloud against mesh (`--no-bake`, wing camera, alpha without dome,
relMSE with it), after coverage was made to follow opacity:

| | alpha before → after (mesh) | relMSE before → after |
|---|---|---|
| sparrow wing, raster | 0.2395 → 0.2367 (0.2253) | 0.1742 → 0.1635 |
| sparrow wing, traced | 0.2385 → 0.2358 (0.2462) | 0.1331 → 0.1247 |

::: {.nota}
**Coverage and opacity** — *coverage* (the image's accumulated alpha) measures what fraction of the pixel the object covers. With `--no-bake` (no baked light) and the camera on the wing, the cloud's mean alpha is compared with the mesh's (in parentheses), together with the relMSE of the colour under a dome. The change "coverage follows opacity" makes a converted gaussian cover exactly what the material's opacity says (rather than treating opacity as glass transmission). It improves both routes: relMSE drops by 6 % and the alpha moves closer to the mesh's on the traced route. That the raster alpha ends above the mesh (0.2367 against 0.2253) and the traced one below (0.2358 against 0.2462) reflects that each route integrates the gaussians differently (projected ellipse versus density along the ray).

*References:* `docs/decisions.md` (sections "Opacity is coverage, not glass", "A converted surface covers what its opacity says").
:::

![Figure 15. The sparrow: the mesh path traced (left) and the converted Gaussians rasterized (right).](figures/img-sparrow.jpg)

For the web, the bird was converted with `--skinned --transfer
--resolution 512`, with zonal lobes baked at 256 paths and 3 bounces in the
bind pose and the 12-part visibility added (`sparrow_tx_vis.usdc`: 1 504 160
gaussians, 1 504 096 with content). Against the mesh path traced at 1024
paths (1024², `air_fly_A0`, frame 16), the cloud's raster render reads
relMSE 0.0025. The cloud's body comes out slightly darker and browner,
because the zonal transfer keeps only direct light.

::: {.nota}
**The conversion for the web** — `--skinned --transfer --resolution 512`: a rigged cloud with zonal lobes baked with 256 paths per gaussian and 3 bounces, in the bind pose (§5.7), plus the 12-part fields (§5.8). 1 504 160 gaussians, of which 1 504 096 have content (the remaining 64 are left without content; the text does not detail the cause). Against the mesh path traced at 1024 spp: relMSE 0.0025, very low; the remaining bias (a darker, browner body) is explained by the zonal transfer storing only the *direct* half: the light bouncing between feathers and from the ground is missing.

*References:* `docs/decisions.md` (sections "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame", "A skinned transfer is traced at the bind pose").
:::

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

::: {.nota}
**The Open Chess Set** — a set of chess pieces from the Academy Software Foundation (ASWF), published as a USD/MaterialX reference, under a CC BY 4.0 licence. The black pawn has two materials: marble (the body) and a glass head (`transmission = 1`, a crystal ball on a gold ring), which also makes it a hard glass test.

**Rigid control** — three invariants: (1) adding a time to `MeshStageOptions` does not change an asset without a skeleton (a byte-identical file: the best non-regression test); (2) the flat-axis change: a gaussian's thickness was $10^{-7}$ in model units, while the other two scales are in cells, so on the fox (a hundred units long) the disc was 1500 times more extreme than on the pawn (66 mm), and the ray tracer, which integrates density along the ray instead of projecting an ellipse, saw a ghost. Making it relative, $\min(\text{along}_U,\text{along}_V)\times0.1$, fixes the fox and barely moves the pawn (changes in the fourth decimal of the marble's mean colour); (3) the skinned route with identities reduces exactly to the rigid one.

*References:* Academy Software Foundation, "The Open Chess Set" (CC BY 4.0); code: `modules/usd/include/athenea/usd/MeshStage.h` (`MeshStageOptions`), `plugins/mesh2splat/mesh2splat.slang`; `docs/decisions.md` (section "The fox", third point "The flat axis is a fraction, not a length").
:::

Its first end-to-end conversion produced 729 073 gaussians at resolution 512.
Its material and transfer results belong to Sections 3 and 4. The
high-quality white-pawn validation (`pawn_mesh.usda`, 2048², 2048 reference
paths) gives, per material:

| material | gaussians | relMSE | p99 | mean (cloud) | mean (path-traced mesh) |
|---|---|---|---|---|---|
| `M_Pawn_Body_W` | 1 568 499 | 0.0218 | 0.273 | 0.167 / 0.165 / 0.133 | 0.164 / 0.162 / 0.132 |
| `M_Pawn_Top_W` (glass) | 307 296 | 0.1418 | 0.354 | 0.236 / 0.260 / 0.221 | 0.365 / 0.397 / 0.279 |

::: {.nota}
**The white pawn's table** — at 2048² and with a 2048-paths-per-pixel reference, the body (1.57 M gaussians) gives relMSE 0.0218 and RGB means within 2 % of the mesh's (0.167 against 0.164 in red); the glass head (0.31 M) gives 0.1418 and a clearly darker mean (0.236 against 0.365): glass, which depends on refraction and reflection of what lies behind it, is the hardest thing to capture in gaussians with low-frequency transfer. p99 is the 99th percentile of the per-pixel error.

*References:* [24], [25], [30] in §10 (glass and microfacets); `docs/decisions.md` (sections "A transparent surface passes what it does not reflect", "A ray that went into a glass meets its far face").
:::

Three later conversions of the same pawn, measured at 1024², give glass
relMSE 0.149, 0.081 and 0.043, with the body unchanged at 0.025. The body is
within a few percent of the path-traced mesh. The error is concentrated in the
glass head (Figure 16).

::: {.nota}
**Evolution of the glass** — three later conversions lowered the glass relMSE from 0.149 to 0.081 and to 0.043 (at 1024²), with the body steady at 0.025: the progress of §4 on glass (Fresnel, the far face, bake rays that leave a lens) is concentrated where the error was.

*References:* `docs/decisions.md` (sections "A transmission pays its Fresnel once", "A glass gaussian's colour is its transmission's own").
:::

![Figure 16. The white pawn at 2048²: the path-traced mesh with 2048 paths per pixel (left) and the converted Gaussians (right).](figures/img-pawn.jpg)

## 5.12 Application: playback on the web and on a phone

The web player (`sparkwebGPU`, a WebGPU fork of Spark) reuses athenea's
kernels compiled to WGSL. athenea's `.athc` writer refuses `--skinned`, so the
web side reads the USD cloud (`usd-athc`, built on the `openusd` crate) and
writes a `.athc` v3. That file adds a `SKIN` section per element, with the
influences packed as joint u16 | weight unorm16 and the gradients as halves,
and an `ATSK` skeleton blob holding joint names, the bind transform and the
first clip. Every other clip is an `ATCL` file fetched when it is played.
Merged LoD nodes carry the opacity-weighted blend of their splats' influences
(the four heaviest, renormalised) and no gradient.

::: {.nota}
**sparkwebGPU and `.athc` v3** — the player is a fork of Spark [11] (World Labs) with a WebGPU backend, which runs athenea kernels compiled from Slang to WGSL [44], [46]. The engine's `.athc` writer does not accept `--skinned`, so the player's `usd-athc` tool (Rust, on the `openusd` crate) reads the USD cloud and writes the `.athc` v3 with: a `SKIN` section per element (influences packed in 32 bits, u16 joint index | unorm16 weight, and gradients as halves), an `ATSK` blob with joint names, the bind transform and the first clip, and `ATCL` files per clip that are downloaded when played.

**unorm16** — a 16-bit unsigned integer interpreted as $x/65535\in[0,1]$: a precision of $1.5\times10^{-5}$, more than enough for weights.

**LoD nodes** — in the player's LoD tree, a node resulting from a merge carries the opacity-weighted blend of its splats' influences (the four heaviest, renormalised, as in the rig mode of §5.9) and no gradients (the Jacobian reduces to the blend term).

*References:* [11], [44], [46] in §10; the sparkwebGPU player (public demos at https://sparkwebgpu.lucab.co.uk).
:::

The skinning kernel (`athenea_adapter/skin.slang`) restates `splatSkin` line
for line, because the original binds eleven storage buffers and Chrome on a
Mac allows ten per stage. A GPU test runs athenea's own kernel on the same
splats and compares the two. Against athenea's raster the result is relMSE
0.0015–0.0053 with a brightness ratio of 1.01. No shutter motion is written.

::: {.nota}
**Why rewrite the kernel** — WebGPU sets per-stage shader limits; the storage-buffer one (`maxStorageBuffersPerShaderStage`) is 8 by default, and Chrome on a Mac exposes 10. `splatSkin` binds 11 (including those of the shutter's second pose and the `motion` buffer). The web version omits motion blur and also takes neither `skelToWorld` nor `worldToPrim`, because the cache's transforms already take skeleton space to cloud space. A GPU test runs athenea's original kernel on the same splats (on a device with the adapter's limits) and compares: relMSE 0.0015–0.0053 against athenea's raster, brightness ratio 1.01.

*References:* [46] in §10 (WebGPU, adapter limits); athenea: `shaders/athenea/scene/splat_skin.slang`.
:::

| page | content | measured |
|---|---|---|
| `/pawn` (published 2026-10-06) | body 727 061 + head 673 711 splats, transfer, 6 CC0 HDRIs | 60 fps in Chrome; v8 data set: download 281.3 → 82.7 MB; desktop GPU memory 1127 → 800 MB |
| `/sparrow` (published 2026-10-07) | 1 504 096 splats, 125.7 MB, 12 clips (6.4 MB gzip), zonal transfer, HDRI | desktop GPU memory 1084 → 684 MB |
| `/sparrow` on phones | `sparrow-mobile`: 488 656 splats, 31.7 MB, cut from the LoD tree (`--keep-splats 600000`) | 60 fps headless in mobile mode, animated |
| sparrow-v2 (candidate) | 1 100 334 splats (−26.8 %), 75.9 MB; mobile 371 138, 26.1 MB | same error as published (4.46e-3) |

::: {.nota}
**The table** — `/pawn`: two clouds (body and head) with transfer and 6 interchangeable CC0 HDRIs; the "v8 dataset" reduced the download from 281.3 to 82.7 MB and the GPU memory from 1127 to 800 MB (the compression of §7). `/sparrow`: 1.5 M splats in 125.7 MB, 12 clips taking only 6.4 MB compressed (the argument of §5.1: clips are matrices). The mobile version is a *cut* of the LoD tree limited to 600 000 splats (`--keep-splats`), 31.7 MB. sparrow-v2 applies the pruning of the last note (−26.8 % of splats) with the same measured error (4.46e-3) as the published one.

*References:* §7 of the paper; public demos `/pawn` and `/sparrow` at https://sparkwebgpu.lucab.co.uk.
:::

**The compact route.** A skinned cloud also plays through the compact path of
§7: each node carries its skin in the compact record, the draw skins it, a
page's box is the union of its boxes over the poses, and the occlusion and
back-face culls are off for what moves. The sparrow's merge tree is split in two
by roots with common zonal scales, two files of about 162 MiB with the clips and
its own domes, 347 MB in all (`sparrow-compact-tx`, published 2026-10-18); on the
desktop it draws 795 k Gaussians at $\tau$ 1.5 and 60 fps while flying, and the
emulated phone 232 k.

::: {.nota}
**The compact route with skinning** — the compact route of §7 stores each tree node in a reduced record and pages it. For an animated cloud: (1) the skin goes in the compact record itself, and drawing applies the skinning; (2) a page's box (the unit of streaming and culling) is the union of its boxes in *all* poses, so that frustum culling stays conservative under motion; (3) occlusion and back-face culling are switched off for what moves (what is hidden or facing away in one pose may not be in another). $\tau$ is the cut's screen-space error threshold (higher = fewer gaussians).

*References:* §7 of the paper.
:::

**Temporal antialiasing.** A temporal resolve was tried in the player (Karis's
scheme: Halton jitter, reprojection, YCoCg clamp and an exponential mean), with
motion vectors per Gaussian, the wings' from the previous frame's bone matrices.
The vectors are right (reprojection error 0.05 → 0.015–0.028 on the wings,
0.018 → 0.003 on the car), but in motion the frame does not beat the frame without
it on the car, the sparrow or the compact path: it shows more error, flicker and
trails, because the highlights are HDR and view-dependent and the jitter is not
averaged out. Variants (Catmull–Rom history, Salvi's variance clip, an $\alpha$
by speed, contrast-adaptive sharpening) do not change that. With the camera still
it matches a 16-frame accumulation. It costs 3 ms at 1080p and is left optional.

::: {.nota}
**Temporal antialiasing (TAA)** — Karis's scheme (2014): every frame the camera is jittered by a sub-pixel offset following a Halton sequence [48]; the previous frame (the *history*) is reprojected with motion vectors; the history is *clamped* to the colour box of the current neighbourhood in YCoCg space (luma/chroma, where the box fits better) to avoid ghosting; and it is blended with an exponential average (typically 10 % of the new frame). Variants tried: sampling the history with a Catmull–Rom filter (sharper than bilinear), Salvi's *variance clipping* (2016: a box at $\mu\pm\gamma\sigma$ of the neighbourhood instead of min/max), a velocity-dependent $\alpha$, and contrast-adaptive sharpening (AMD CAS style).

**Why it does not win here** — the motion vectors are correct (very low reprojection error, also on the wings with the previous frame's bone matrices), but the highlights of relit gaussians are HDR and view-dependent: the clamp either clips them or lets them flicker, and the jitter does not average out under motion. With the camera still it converges to a 16-frame accumulation. It remains an option (3 ms at 1080p).

*References:* [48] in §10 (Halton); Karis (2014), "High Quality Temporal Supersampling", SIGGRAPH Advances in Real-Time Rendering; Salvi (2016), "An Excursion in Temporal Supersampling", GDC.
:::

**Pruning a moving asset.** Gaussians hidden in the bind pose cannot simply be
dropped, because other poses reveal them. Measured over 12 clips × 4 samples
= 48 poses plus the bind pose, 61 views each: 61.2 % of the sparrow is hidden
in the bind pose, but only **32.5 %** is hidden in every pose. Removing what
the bind pose hides costs relMSE up to $4.5\times10^{-2}$ in close views of
unsampled poses. Removing what every pose hides costs at most
$2.9\times10^{-4}$. sparrow-v2 is the latter. For comparison, the static pawn
has only 2.7 % hidden.

::: {.nota}
**Pruning with poses** — a fully hidden gaussian (covered by others from every view) does not contribute to the image and can be removed, but in an animated asset it must be hidden in *every* pose. Sampling: 12 clips $\times$ 4 instants + bind = 49 poses, 61 views each. In the bind pose 61.2 % is hidden (the inside of the body, feathers under other feathers), but only 32.5 % is hidden in every pose: a wing that opens uncovers what it was hiding. Removing what is hidden only in the bind pose costs a relMSE of up to $4.5\times10^{-2}$ in close views of unsampled poses; removing what is hidden in all of them, at most $2.9\times10^{-4}$. The static pawn, with hardly any occluded interior, has only 2.7 % hidden.

**Risk** — it is an *empirical* pruning: an unsampled pose could reveal something removed; that is why the error is measured on poses that took no part in the selection.

*References:* [10] in §10 (GaussianPOP, error-based pruning).
:::

## 5.13 What is not done

- More than four influences per gaussian. This limits accuracy at
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

::: {.nota}
**What is missing, grouped** — (1) *Rig fidelity*: more than four influences, DQS for clouds, several meshes with different prim transforms, skeletons with an animated transform of their own. (2) *Animation*: retargeting and clip blending (a cloud with transforms baked into the primvar has no local joints to blend; it would have to go through the Skeleton). (3) *Motion*: the traced route poses once per frame (no skinning motion blur), and LOD instances carry no displacement. (4) *Shadow*: intra-part self-shadowing (the literature proposes canonical AO per bone [42] or a short trace), inter-part occlusion inside the zonal transfer in other poses, and bakes above about 3 M gaussians per part (the BLAS memory limit). (5) *Measurements*: the zonal fit error on the sparrow, and a geometric coverage measurement on a close-up bend, which is where the gain of the elastic term would show.

*References:* [42] in §10; `docs/decisions.md` ("Not done" paragraphs of the sections "And the engine puts the cloud where the skeleton is", "The whole Jacobian of a skinned gaussian", "A skinned transfer is traced at the bind pose", "A cloud shadows itself by part…").
:::
