<!-- Copyright (c) 2026 jesus luque. -->

# 6. Levels of detail: merge trees, cuts by error, pages and occlusion

A cloud baked from a mesh at 1280 px holds 8.7 million Gaussians for one car. The close-up level baked at 5120 px
holds 74.5 million. Views range from the whole asset far away to a badge close up, and a cloud of that size can be
neither drawn nor kept resident whole: what is drawn and what is resident must follow the view, under a node budget
and a memory budget, with the data streamed in units. This section first describes the engine's original level of
detail (§6.1), which was built for captured scenes, and explains why it does not scale to such clouds. It then
covers the structure that replaces it: per-part pair-merge trees (§6.2), one cut by a bounded screen error (§6.3), the
opacity a merge is given (§6.4) and a spill metric that was evaluated and not adopted (§6.5), two view-dependent culls
(§6.6, §6.7), pages ordered by error (§6.8), the 5120 px close-up level (§6.9) and the measurements (§6.10). Every
kernel named here is a Slang compute kernel. The CPU reads files and does bookkeeping only (§2).

::: {.nota}
**Level of detail (LOD)** — A family of techniques that replaces one representation with a cheaper one when the difference is not perceptible from the current view. For meshes, the tradition runs from discrete levels (several meshes per object, chosen by distance) to continuous or hierarchical representations (Hoppe's *progressive meshes*, 1996; simplification by *quadric error metrics*, Garland and Heckbert, 1997; the standard reference is Luebke et al., *Level of Detail for 3D Graphics*, 2002). In Gaussian splatting the problem is the same, but the primitive is a 3D Gaussian with an opacity: replacing $k$ Gaussians with one forces a decision on which Gaussian "represents" the group (here, by *moment matching*, §6.1–6.2) and on when it is acceptable to draw it (here, by an error projected in pixels, §6.3).

**Why it is needed here** — 8.7 M Gaussians per car at 1280 px and 74.5 M at 5120 px exceed two different limits: the drawing limit (the rasteriser sorts and composites every Gaussian in every tile it touches, at a cost roughly linear in the number of visible Gaussians) and the memory limit (each Gaussian with a TX transfer takes hundreds of bytes; the 148.9 M nodes of the close-up level take 87.4 GB on disk, §6.9). That is why the text separates two *budgets*: one of **nodes** drawn per frame (render cost) and one of resident **memory** (what fits on the device or on a phone). What is resident is loaded by **streaming**, that is, in units (chunks in §6.1, pages in §6.8) fetched on demand according to the view.

**Cut** — The central term of the section: in a tree of levels, a *cut* is a set of nodes such that every leaf has exactly one ancestor (or itself) in the set. Drawing a cut amounts to drawing the whole cloud at a resolution that varies by region. The following subsections build the tree (§6.2), choose the cut (§6.3), and decide which parts of the tree must be loaded in order to choose it (§6.8).

**Alternatives in the literature** — *Hierarchical 3D Gaussians* (H3DGS, Kerbl et al. 2024, [2]) builds a hierarchy over a captured scene and picks a cut by projected granularity; *Octree-GS* ([8]) organises anchors into octree levels chosen by distance; LODGE (Kulhánek et al., 2025) combines distance-based levels with chunk partitioning for large scenes on modest hardware; Nanite (Karis et al. 2021, [35]) uses a DAG of triangle clusters with monotone error. The design of this section takes the moment representation from [2], pair merging from NanoGS ([9]), and from Nanite the parallel selection by monotone error and paging by error.

**The CPU rule** — "The CPU reads files and does bookkeeping only" is the engine's rule (the repository's `CLAUDE.md`: *No CPU arithmetic on data*): sorting, merging, culling and image comparison are compute kernels; the host reads only counters and histograms.

*References:* [2], [8], [9], [35] in §10; Hoppe, H. (1996) *Progressive Meshes*, SIGGRAPH; Garland, M., Heckbert, P. (1997) *Surface Simplification Using Quadric Error Metrics*, SIGGRAPH; code: `modules/lod/src/Lod.cpp`, `shaders/athenea/lod/`; `docs/decisions.md` (sections "Levels of detail", "Decimation: the levels of detail, cut once by what a merge loses", "Simplify: fewer gaussians by pairs, and surfels").
:::

Notation. A node $n$ is a leaf (a baked Gaussian) or a merge. $\mu_n$ is its centre and $\Sigma_n$ its covariance,
with scales $s_n = (s_x, s_y, s_z)$ and opacity $o_n$. $\mathrm{par}(n)$ is its parent. $\tau$ is the threshold in
pixels, $f$ is the focal length in pixels at unit distance (`pixelsPerUnit`), and $e$ is the eye.

::: {.nota}
**3D Gaussian** — Each splat is a density $G(x)=\exp\!\big(-\tfrac12 (x-\mu)^\top\Sigma^{-1}(x-\mu)\big)$ multiplied by an opacity $o\in[0,1]$. The covariance factorises as $\Sigma = R\,\mathrm{diag}(s_x^2,s_y^2,s_z^2)\,R^\top$: the **scales** $s$ are the standard deviations along the three principal axes (the columns of $R$, stored as a quaternion). A scale of exactly 0 is a **surfel** (a 2D disc, [5], [6]).

**Leaf and merge** — A *leaf* is a Gaussian produced by the bake (§3–4); a *merge* is an internal node of the tree that stands for all the leaves under it. $\mathrm{par}(n)$ is its parent in the tree; in the file a root is its own parent (`lodParent`).

**$f$ (`pixelsPerUnit`)** — The focal length expressed in pixels: for a perspective camera with image height $H$ and vertical field of view $\mathrm{fov}_y$, $f = \frac{H/2}{\tan(\mathrm{fov}_y/2)}$. An object of size $L$ at distance $d$ covers roughly $L f/d$ pixels. The whole section converts quantities in metres to pixels with that ratio. In the code it is `pixelsPerUnit` in `LodParams` (`shaders/athenea/lod/lod_common.slang`) and in `CutParams` (`shaders/athenea/lod/lod_tree_cut.slang`), and it also multiplies in the scale of the instance's transform.

**$\tau$** — Threshold in pixels: the largest projected error (or size) accepted for drawing a node instead of its children. $\tau=0.25$ px means the error of using the merge is a quarter of an output pixel.

*References:* [1], [3], [5], [6] in §10; code: `shaders/athenea/lod/lod_common.slang` (`LodParams`), `shaders/athenea/lod/lod_blob.slang` (`Blob`: centre, opacity, scales, axes as rows, colour).
:::

## 6.1 The engine's original level of detail

**Octree by Morton code.** `LodBuilder` (`modules/lod`, `shaders/athenea/lod/lod_common.slang`) sorts the splats by a
30-bit Morton code. The octree cell of a splat at level $r$ is the top $3r$ bits of its code, so the cells of a level
are runs of equal prefixes. A boundary pass and a prefix sum find them, and no tree is walked. Each cell carries
moments, following Kerbl et al. 2024. With $w = o \cdot A_2(s)$, where $A_2$ is the area of the two longest axes,

$$W=\sum w,\quad S_1=\sum w\,\mu,\quad S_2=\sum w\,(\Sigma+\mu\mu^\top),\quad C=\sum w\,c .$$

::: {.nota}
**Morton code (Z-order)** — Each coordinate of the centre, normalised to the bounding box, is quantised to 10 bits, and the bits are interleaved ($x_9y_9z_9x_8y_8z_8\ldots$) into a 30-bit integer (Morton, 1966). Sorting by that code walks space along a Z-shaped space-filling curve, so points close in the order tend to be close in space. The property used here is stronger: the top $3r$ bits are exactly the index of the cell of a level-$r$ **octree** (with $2^r$ cells per axis). So, after sorting, the Gaussians of one cell form a contiguous run, and each octree level is a partition of the same sorted array into runs.

**Boundary pass and prefix sum** — For level $r$, a kernel sets $b_i = 1$ if the $3r$-bit prefix of element $i$ differs from that of $i-1$ (`lodBoundaries`); an exclusive prefix sum of $b$ gives each element the index of its group and the total number of groups. This is Karras's octree construction ([32]): the whole tree comes out of one sort and elementwise parallel operations, with no pointers and no traversals.

**Moments of a group** — With a weight $w$ per Gaussian, the zeroth-order moment ($W$, the mass), the first-order moment ($S_1$, mass times centre) and the second-order moment ($S_2$) are accumulated. The term $\Sigma+\mu\mu^\top$ appears because the second moment of a Gaussian about the origin is $E[xx^\top]=\Sigma+\mu\mu^\top$. $C$ is the weighted colour (and in the code also the SH coefficients, the normals and the emission, `kMomentsHead` in `lod_common.slang`).

**The weight $w = o\cdot A_2(s)$** — What a Gaussian contributes to the image is roughly its opacity times the area it covers. For a surface (discs) the relevant area is that of the two longest axes; using the volume ($s_xs_ys_z$) would give a surfel zero weight. In the code it is `discArea`, $\max(s_xs_y, s_ys_z, s_xs_z)$, the product of the two largest scales.

**A note on the origin** — The formula for $S_2$ written here is about the origin, as in [2]. The current code takes the second moment about the group's own mean (§6.2.1 explains the numerical reason): `lod_common.slang` defines $S_2=\sum w\,(\Sigma+(\mu-m)(\mu-m)^\top)$ with $m=S_1/W$.

*References:* [2], [32] in §10; Morton, G. M. (1966) *A computer oriented geodetic data base and a new technique in file sequencing*, IBM; code: `shaders/athenea/lod/lod_morton.slang` (`lodMorton`), `shaders/athenea/common/morton.slang` (`morton30`), `shaders/athenea/lod/lod_boundaries.slang` (`lodBoundaries`), `shaders/athenea/lod/lod_leaf_moments.slang` (`lodLeafMoments`), `modules/gpu/include/athenea/gpu/algo/PrefixSum.h`, `RadixSort.h`; `docs/decisions.md` ("Levels of detail" → "Built and cut on the device").
:::

Moments add, so each coarser level is summed from its children. A cell becomes one Gaussian with $\mu=S_1/W$,
$\Sigma=S_2/W-\mu\mu^\top$ (Jacobi sweeps in the shader), $c=C/W$ and $o=\min(W/A_2,0.99)$. The stored levels run from
`coarsestLevel` down to the deepest level whose cell count is at most half the splat count.

::: {.nota}
**Moment matching** — Given a mixture of Gaussians with weights $w_i$, the only Gaussian that preserves its mass, mean and covariance is $\mu=S_1/W$, $\Sigma=S_2/W-\mu\mu^\top$. It is also the Gaussian that minimises $\mathrm{KL}(p\,\Vert\,q)$ between the mixture $p$ and a Gaussian $q$ (the "M-projection" of information theory; see Runnalls [34]). That moments add is what allows the levels to be built bottom-up: a parent's moments are the sum of its children's (`lodMergeMoments`), and the result does not depend on how the sums are grouped.

**Jacobi sweeps** — Going from $\Sigma$ to (rotation, scales) requires diagonalising a symmetric $3\times3$ matrix. The Jacobi method applies plane rotations that zero each off-diagonal element in turn; for $3\times3$ it converges in a few sweeps and is implemented branch-free in a shader (function `jacobi` in `lod_finalize.slang` and in `lod_pairs_final.slang`). The scales are the square roots of the eigenvalues; the rotation, the eigenvectors.

**Opacity $o=\min(W/A_2, 0.99)$** — The mass divided by the merge's own area gives the fraction of that area that the leaves "covered". The 0.99 cap is the 3DGS convention ([1]), which limits $\alpha$ so that the accumulated transmittance never reaches zero. §6.4 shows that this estimator double-counts the regions where the leaves overlap.

**Which levels are stored** — `coarsestLevel` (1 by default in `LodBuildSettings`, `modules/lod/include/athenea/lod/Lod.h`) is the coarsest level. The finest level stored is the deepest whose cell count is at most half the splat count: below that a level hardly reduces anything (most cells would hold a single Gaussian), and it is not worth the memory.

*References:* [1], [2], [34] in §10; Golub, G., Van Loan, C. *Matrix Computations*, ch. 8 (the Jacobi method); code: `shaders/athenea/lod/lod_merge_moments.slang` (`lodMergeMoments`), `shaders/athenea/lod/lod_finalize.slang` (`lodFinalize`, `jacobi`), `modules/lod/src/Lod.cpp` (`LodBuilder`).
:::

**Cut.** `CutSelector` draws a group when its cell's edge, projected at the cell's nearest distance, is at most $\tau$
and its parent cell's projection is above $\tau$. A child cell lies inside its parent, so the test is monotone along
any root-to-leaf line, and every place is drawn at exactly one level. The only readback is a count per level and a
read per instance per frame.

::: {.nota}
**Cut by size in the octree** — For each group, `projectedEdge` (`shaders/athenea/lod/lod_cut.slang`) computes its cell's edge, $\ell_r = \text{extent}/2^r$, times $f$, divided by the distance from the eye to the nearest point of the cell (the eye is projected onto the box with a `clamp`). If the eye is inside, the distance is almost zero and the cell is never "small". The group is drawn if it is small and its parent is not (`lodCutGroups`).

**Monotonicity** — A child cell is contained in its parent: its nearest point to the eye is at a greater or equal distance and its edge is half as long. Its projected size is therefore smaller or equal. Along any path from the root to a leaf, the predicate "small" switches from false to true exactly once. The node where it switches is the only one drawn on that path: that is what "every place is drawn at exactly one level" means, and it is the same proof structure as the proposition of §6.3.

**Readbacks** — Selection is one kernel per level, fully parallel; only counters return to the CPU (how many groups each level selects, to size the gather), in a single read per instance per frame.

*References:* [2], [32] in §10; code: `shaders/athenea/lod/lod_cut.slang` (`projectedEdge`, `small`, `lodCutGroups`, `lodCutFinest`, `lodCutSplats`), `modules/lod/src/Lod.cpp` (`CutSelector`); `docs/decisions.md` ("Levels of detail" → "Built and cut on the device").
:::

**Chunks, `.athc` and streaming.** The splats are cut into chunks of 65 536 consecutive Morton-ordered splats. The
merged levels stay on the device, and the chunks are streamed. The `.athc` format (`.lrtc` in the lucabRTrender tree)
has a 104 B header, level and chunk tables, and blocks aligned to 4 KiB so that a chunk can be mapped and faulted in
alone. `StreamingPool` maps the file. Two loader threads copy the wanted chunks, most wanted first. Each slot is taken
from a free slot, from the chunk wanted least recently, or from a chunk wanted less than half as much. A finest-level
group whose chunks are missing draws its own merged Gaussian, so the image never has a hole. On train_7k (742 k
splats, M5 Pro), the build took 47 ms and produced 146 k merged Gaussians. At 8 px the cut drew 83 k splats (11 %),
rendering in 3.3 ms against 12.6 ms, with a mean absolute error of $3\cdot10^{-3}$. On train_30k, the `.athc` streamed
into 4 of 16 slots wrote byte-identical EXRs and settled in 2 cuts.

::: {.nota}
**Chunks** — Fixed runs of 65 536 ($2^{16}$, `chunkSplats` in `LodBuildSettings`) consecutive Gaussians in Morton order: thanks to the locality of the Z-order, each chunk is a compact region of space. The fixed size makes every slot of the device store the same size and avoids fragmentation.

**`.athc`** — A binary container: a 104 B header (`static_assert(sizeof(FileHeader) == 104)` in `modules/lod/src/Athc.cpp`), level and chunk tables, and blocks aligned to 4 KiB, the usual virtual-memory page size. The alignment lets the file be mapped (`mmap`) so that touching a chunk produces only that chunk's page faults. The bytes use the same packing as on the device, so loading is copying. The writer writes `name.partial` and renames it, so that a failure never leaves a file that looks complete.

**`StreamingPool`** — Maps the file; two loader threads (`loaders = 2` in `modules/lod/include/athenea/lod/Athc.h`) copy the wanted chunks from the mapping, so page faults happen off the frame thread. The replacement policy, in order: a free slot; that of the chunk not wanted now that was wanted longest ago (LRU); or that of a chunk wanted less than half as much as the incoming one. That factor of 1/2 is a **hysteresis**: it prevents two chunks of similar need from evicting each other on alternate frames.

**No holes** — A group of the finest merged level whose chunks are missing draws its own merged Gaussian (the nearest resident ancestor), so the image degrades in resolution, not in coverage.

**The figures** — *train* is the Tanks and Temples scene used in 3DGS ([1]); `_7k` and `_30k` are the checkpoints at 7 000 and 30 000 training iterations. 47 ms to build for 742 k splats and 146 k merged Gaussians (an extra 20 %). At 8 px the cut draws 83 k (11 %) and the render drops from 12.6 to 3.3 ms with a mean absolute error of $3\cdot10^{-3}$ (on colour values in $[0,1]$). "Settled in 2 cuts": after two frames of cut and loading, no further chunk changes state. The byte-identical EXRs show that streaming does not alter the result once settled.

*References:* [1], [2] in §10; code: `modules/lod/include/athenea/lod/Athc.h` (`StreamingPool`), `modules/lod/src/Athc.cpp`, `shaders/athenea/lod/lod_cut.slang` (`lodChunkNeeds`, `lodCutFinest`); `docs/decisions.md` ("Levels of detail" → "Chunks, `.athc` and streaming", "Measured: streaming (M5 Pro)", "In USD: `AtheneaStreamedAssetAPI`"); `docs/operations.md` §3.3 "A large cloud: levels of detail and streaming".
:::

**Why it does not scale to baked assets.** Four properties of this structure do not fit clouds of tens of millions
of material Gaussians under a shared budget.

1. The threshold is set per instance and is not a budget. Each instance costs two blocking submits and a readback per
   frame, and no budget is shared across instances.
2. The octree cells ignore the object. A cell merges whatever falls inside it, including a badge with the panel
   beneath it and glass with paint.
3. A streamed splat costs 132 B, against ~55 B for the compact record (§7.4).
4. Only the splats are chunked. The merged levels, down to one with up to half as many cells as splats, are always
   resident, and for millions of leaves they alone exceed the budget.

::: {.nota}
**1. Threshold per instance** — In the original design each `LodInstance` carries its own threshold, and each instance makes two dispatches with a wait (counters and gather) and one readback. There is no global number of Gaussians to share out among instances: one cannot say "draw at most 250 k in total". The error cut of §6.3 does admit a global budget (`--cut-budget`, bisection on $\tau$).

**2. Cells that ignore the object** — An octree cell is a box in space: it merges whatever falls inside without looking at material or part. In a baked material cloud that mixes glass with paint, a badge with the panel beneath it, or the two faces of a door. The merge tree of §6.2 solves this with guards on part key and material.

**3. 132 B per splat** — The streaming format stores the device packing: float4 position (16 B), shape (4 uint, 16 B), SH coefficients in f16 and the group index, per splat. The player's compact record (§7.4) quantises and compresses down to about 55 B: with tens of millions of splats, the difference is gigabytes.

**4. Levels always resident** — Only the splats are chunked; the merged levels (up to $n/2$ cells in the finest) are uploaded whole when the file is opened. For 74.5 M leaves that would be up to ~37 M merged Gaussians always in memory, more than a phone's whole budget. The pages by error of §6.8 make the internal nodes paged as well.

*References:* [2] in §10; code: `modules/lod/include/athenea/lod/Lod.h` (`LodInstance`, `CutSelector`), `modules/lod/src/Athc.cpp`; `docs/decisions.md` ("Levels of detail" → "Not done yet").
:::

## 6.2 Per-part pair-merge trees

`athenea decimate --tree` builds a binary merge tree by mutual nearest-neighbour pair merging. It is NanoGS (Xiong et
al. 2026) on the device (`lod_pairs.slang`, `lod_pairs_final.slang`, `Lod.cpp`), with guards that suit baked material
clouds (Figure 17).

::: {.nota}
**Binary merge tree** — A tree in which every internal node is the union of exactly two nodes. It is the result of **agglomerative hierarchical clustering**: start with each Gaussian as a cluster and repeatedly join pairs of clusters that are "cheap to join". The tree records the order of the joins, and any cut of that tree is a valid simplification of the cloud.

**Mutual nearest neighbours** — Each node picks the neighbour with the lowest merge cost; a pair $(a,b)$ merges only if $a$ picks $b$ and $b$ picks $a$. This condition allows many merges in parallel without conflicts (each node is in at most one pair) and, for costs that satisfy *reducibility* (the cost to a joined cluster is no lower than the minimum to its parts), yields the same tree as the sequential greedy agglomeration (Walter et al. 2008, *Fast Agglomerative Clustering for Rendering*; the principle goes back to the nearest-neighbour chain of Benzécri/Juan, 1982). With costs that do not satisfy it exactly, as here, it is a parallel approximation.

**NanoGS** ([9]) proposes this mutual pair merging for Gaussians; athenea moves it onto the device (`lod_pairs.slang`: `pairCodes`, `pairBest`, `pairHistogram`, `pairMerge`; `lod_pairs_final.slang`: `pairTreeLayout`, `pairFinalize`) and adds guards for material clouds (§6.2.3). `athenea decimate --tree` records the merges as a tree instead of keeping only the result at the requested ratio.

**Against the alternative** — The octree of §6.1 groups by position; the merge tree groups by cost, which can include colour, normal, material and part. In exchange, the tree has no regular levels: a cut can hold 1-leaf nodes next to nodes of $2^{11}$ (§6.8).

*References:* [9] in §10; Walter, B., Bala, K., Kulkarni, M., Pingali, K. (2008) *Fast Agglomerative Clustering for Rendering*, IEEE Symposium on Interactive Ray Tracing; code: `shaders/athenea/lod/lod_pairs.slang`, `shaders/athenea/lod/lod_pairs_final.slang`, `modules/lod/src/Lod.cpp` (`Decimator`, `PairSettings`); `apps/athenea/src/CmdDecimate.cpp` (`--tree`); `docs/decisions.md` ("Simplify: fewer gaussians by pairs, and surfels").
:::

![Figure 17. A merge tree and its cut. Nodes are sums of moments; the error ε is monotone up the tree and each node’s sphere ρ covers its children’s. Nodes are paged by their parent’s error and carry an index into a conservative occlusion codebook; the per-frame traversal tests pages, back faces and occlusion.](figures/fig-lod.svg)

::: {.nota}
**Reading Figure 17** — Nodes are sums of moments (§6.2.1). The error $\varepsilon$ is "monotone up the tree": it never decreases from a child to its parent (§6.3). Each node's sphere $\rho$ contains its children's, so the distance from the eye to the sphere does not grow going up either. These two properties guarantee that the cut is a local test per node. Pages (§6.8) group nodes by their parent's error, and each node carries an index into a codebook of occlusion masks (§6.7). A frame's traversal discards whole pages (by error projected from their box), back faces (§6.6) and occluded nodes (§6.7).

*References:* [35] in §10 (the same combination in Nanite: monotone error, covering spheres and streaming by pages); code: `shaders/athenea/lod/lod_tree_cut.slang`, `shaders/athenea/lod/lod_pages.slang`, `shaders/athenea/lod/occlusion_mask.slang`.
:::

### 6.2.1 Nodes as moments

A node is a vector of 24 floats: $W$, $W\mu$, $W(\Sigma+(\mu-\bar\mu)(\mu-\bar\mu)^\top)$ (6), $W\cdot\mathrm{sRGB}(c)$, $W n$, $W\alpha$,
$W\cdot$metallic, $W\cdot$transmission, $W\cdot$specular, the key, the record count, the weighted thickness ratio and
the node's index in the tree. Here $w = o\cdot\max(s_xs_y, s_ys_z, s_xs_z)$ (`discArea`), and $\alpha$ is the GGX
roughness squared. A merge sums the two vectors. Because sums are associative, a node's moments are those of all its
leaves whatever the merge order.

::: {.nota}
**The 24-float vector** — It is `kNodeFloats` in `lod_pairs.slang`, laid out as: 0 $W$ | 1–3 $W\mu$ | 4–9 $W\Sigma$ about the mean (the six unique components of a symmetric matrix: xx xy xz yy yz zz) | 10–12 $W\cdot$sRGB colour | 13–15 $W\cdot$normal | 16 $W\alpha$ | 17 metallic | 18 transmission | 19 specular | 20 key (bits) | 21 record count | 22 weighted thickness | 23 node index in the tree. Everything the cost compares is kept as a mass-weighted sum, so a node's mean is the sum divided by $W$.

**Colour in sRGB** — Colour is accumulated sRGB-encoded rather than linear, so that a dark-on-light difference counts as it is perceived (sRGB encoding is roughly perceptually uniform; in linear, the darks are compressed near 0).

**$\alpha$ = roughness$^2$** — The usual GGX parametrisation (Walter et al. [24], Burley [27]): the angular width of the specular lobe grows roughly with $\alpha$.

**Associativity** — Because a merge is a sum of vectors, the root node of a subtree has the same moments whatever the order and shape of the merges inside it: the node "is" the moment-matched Gaussian of all its leaves.

**Attributes merged at the end** — TX transfer (tens of floats per Gaussian), shadow bits, material, a rig's joints: these are not carried through every pass but merged once, mass-weighted over the node's run of leaves (`lodMergeAttribute` in `lod_attributes.slang`). Because a node's leaves are a contiguous run in depth-first order (§6.2.4, "Layout"), the kernel walks `[lodStart, lodStart + lodSize)`.

*References:* [24], [27] in §10; code: `shaders/athenea/lod/lod_pairs.slang` (`kNodeFloats`, `Node`, `nodeOf`), `shaders/athenea/lod/lod_attributes.slang` (`lodMergeAttribute`, `lodAttributeKey`), `shaders/athenea/lod/lod_pairs_final.slang` (`discArea` via `lod_blob`); `docs/decisions.md` ("Simplify: fewer gaussians by pairs, and surfels").
:::

The second moment is taken about the group's own centre $\bar\mu$, not the origin: a leaf group's from each Gaussian's
offset to the group's mean, a coarser group's from its children's with each child's mean offset (the parallel axis
theorem), a pair's merge the same way. About the origin, $\Sigma = S_2/W - \mu\mu^\top$ is the difference of two
numbers the size of $\mu\mu^\top$, which a float holds to about $5\times10^{-7}$ m² two metres from the origin --
the whole variance of a Gaussian a millimetre wide. The close-up bakes' needles on long thin triangles (0.93 × 0.043
× 0.004 mm on the front bumper, variances across of $2\times10^{-9}$ m²) merged as thin as the rounding left them, and
the cost read from the same moments called them nearly free, so a cut chose them and drew their holes as bright grain.
On the bumper's front-left corner baked alone at 5120 px (301 k Gaussians), the cut at 0.01 px from a close camera went
from relMSE $8.8\times10^{-4}$ to $1.3\times10^{-5}$ against the bake, and the cut at 0.25 px from `/World/Camera`
draws 42 719 Gaussians instead of 94 351 at slightly less error ($3.7$ against $3.9\times10^{-4}$). Every other attribute (transfer, shadow bits, material, joints) is merged once at the
end, mass-weighted over the node's run of leaves (`lodMergeAttribute`).

::: {.nota}
**Catastrophic cancellation** — A 32-bit float has a 24-bit mantissa: its relative precision is $2^{-24}\approx6\times10^{-8}$. Two metres from the origin, $\mu\mu^\top$ is of the order of $4$ m², and the spacing between consecutive floats near 4 is $2^{2}\cdot2^{-23}\approx4.8\times10^{-7}$ m²: the "$5\times10^{-7}$" of the text. Subtracting $S_2/W-\mu\mu^\top$ with both terms that large leaves a result with an absolute error of that order. A Gaussian 1 mm wide has a variance of the order of $(0.5\,\text{mm})^2=2.5\times10^{-7}$ m²: the rounding swallows it whole. It is the textbook example in Goldberg (1991).

**Parallel axis theorem (Steiner)** — If each child $k$ has mass $W_k$, mean $\mu_k$ and centred second moment $M_k$, the parent's second moment about its mean $\bar\mu$ is
$$M_p=\sum_k\big(M_k + W_k(\mu_k-\bar\mu)(\mu_k-\bar\mu)^\top\big).$$
For a pair it simplifies to $M_a+M_b+\frac{W_aW_b}{W_a+W_b}\,d\,d^\top$ with $d=\mu_a-\mu_b$ (the `between` term of `pairMerge`). The offsets are millimetres, and what a float loses on them is small in relative terms too. It is the same idea as Welford's algorithm for online variance.

**The needles** — 0.93 × 0.043 × 0.004 mm: very elongated Gaussians that the 5120 px bake lays on long thin triangles of the bumper. The variance across, $(0.043\,\text{mm})^2\approx1.8\times10^{-9}$ m², is about 250 times smaller than the rounding, so the merge came out as wide as the rounding dictated, and the shape cost (computed from those same moments) was almost zero: the cut preferred them and drew holes.

**relMSE** — Relative mean squared error, $\frac1N\sum\frac{(x-y)^2}{y^2+\epsilon}$, the usual metric in Monte Carlo rendering because bright regions do not dominate it. 8.8e-4 → 1.3e-5 is almost two orders of magnitude. In the far view the corrected tree draws less than half as many Gaussians (42 719 against 94 351) with slightly lower error: the wrong costs kept valid wide merges apart and let degenerate merges through.

*References:* Goldberg, D. (1991) *What Every Computer Scientist Should Know About Floating-Point Arithmetic*, ACM Computing Surveys; Welford, B. P. (1962) *Note on a method for calculating corrected sums of squares and products*, Technometrics; code: `shaders/athenea/lod/lod_pairs.slang` (`pairMerge`), `shaders/athenea/lod/lod_merge_moments.slang`, `shaders/athenea/lod/lod_leaf_moments.slang`, `shaders/athenea/lod/lod_pairs_final.slang` (`pairFinalize`, moments about the first member); `docs/decisions.md` ("A merge's second moment about its own centre (13 Oct)").
:::

The merged Gaussian (`pairFinalize`) takes the moment-matched $\mu$ and $\Sigma$, eigen-decomposed by Jacobi with the
thinnest axis third. Merging by moments preserves the second moment. A $2\times2$ block of discs as wide as their
spacing therefore merges into one disc 0.58 of the spacing wide, which leaves visible holes when seen up close.
Along each wide axis $u$, the merge is widened as follows. Let $v_u$ be the area-weighted variance of the member
centres, $c_u$ their mean width, and $d^2 = 12\sqrt{v_uv_v}/(n-1)$ the spacing of a grid of $n$ members over that
spread. Then

$$s_u' = \mathrm{clamp}\!\left(\frac{c_u}{d}\sqrt{12v_u+d^2},\; s_u,\; 2s_u\right).$$

::: {.nota}
**Why moment matching narrows** — The second moment of a uniformly covered patch of side $L$ is $L^2/12$ per axis, so the moment-matched Gaussian has $\sigma=L/\sqrt{12}\approx0.29L$. A $2\times2$ block of discs spaced $D$ apart covers a side $L=2D$; its merge has $\sigma\approx2D/\sqrt{12}=D/\sqrt3\approx0.58D$: the figure in the text. But a Gaussian of that $\sigma$ falls to 60 % of its peak at $1\sigma$ and to 14 % at $2\sigma$: at the block's edge (at $D$ from the centre, $\approx1.7\sigma$) it covers considerably less than the four original discs. Neighbouring merges like this leave a grid of holes visible up close (on the test floor, p99 70 unwidened against 1 with the widening, `docs/decisions.md`).

**The widening formula** — Along each wide axis $u$:
- $v_u$: area-weighted variance of the member centres along $u$.
- $d$: spacing of a regular grid of $n$ points with that spread. For a $k\times k$ grid ($n=k^2$) of step $d$, the per-axis variance of the centres is $d^2(k^2-1)/12=d^2(n-1)/12$; solving, $d^2=12v/(n-1)$. With unequal axes the geometric mean $\sqrt{v_uv_v}$ is used.
- $\sqrt{12v_u+d^2}$: the actual length the row covers, $L=kd$, because $12v+d^2=d^2(k^2-1)+d^2=k^2d^2$.
- $c_u/d$: the members' width-to-spacing ratio. The merge keeps that ratio relative to the total span: $s_u'=(c_u/d)\,L$.

The `clamp` to $[s_u,2s_u]$ prevents narrowing below the moment and limits the degenerate case of nearly collinear members, where $d\to0$ and the widening would blow up (the merge would turn into a stroke).

**Decomposition** — `pairFinalize` diagonalises with Jacobi and reorders so that the thinnest axis is third (`thinAxisThird` in `shaders/athenea/common/packing.slang`); the disc's normal is oriented towards the side of the members' shading normals.

*References:* code: `shaders/athenea/lod/lod_pairs_final.slang` (`pairFinalize`, `jacobi`), `shaders/athenea/lod/lod_decimate.slang` (`lodDecimateShape`, the same rule in octree decimation), `shaders/athenea/common/packing.slang` (`thinAxisThird`); `docs/decisions.md` ("Decimation: the levels of detail, cut once by what a merge loses" → "Merged wider than the moments"; "Simplify…" → "Coverage").
:::

The opacity under the default rule is the members' area-weighted mean, capped at 0.99. The mass coverage $W/A(s')$ is
kept as `coverageMass`. A merge of discs whose thinnest axis is under 0.12 of its middle axis is written as a surfel
(thinnest scale exactly 0).

::: {.nota}
**Merge opacity (default rule)** — The members' opacities averaged with area weights ($\sum A_l o_l/\sum A_l$), capped at 0.99. With the widening above, the merge covers roughly what its members covered, so inheriting their mean opacity is consistent. The "mass over area" alternative measured worse in decimation (the Corvette's paint smeared past its edges, because a sum of overlapping discs is not their union).

**`coverageMass`** — $W/A(s')$: the mass divided by the area of the already widened merge, without the 0.99 cap. It is stored as a primvar (`primvars:athenea:splat:coverageMass`) because the web player uses it as opacity (§6.4, end).

**Surfel** — If all members are discs and the merge comes out with its thinnest axis below 0.12 of the middle one (`flatTolerance = 0.12` in `PairSettings`), it is written with that scale exactly 0: a 2D disc, as in 2D Gaussian Splatting ([5]) and in the surfels of Pfister et al. ([6]). The rasteriser draws it by ray-plane intersection at each pixel; the f16 stores the zero scale as a very negative logarithm, never $-\infty$.

*References:* [5], [6] in §10; code: `shaders/athenea/lod/lod_pairs_final.slang` (`pairFinalize`, `outCoverage`, `surfelCount`), `modules/lod/include/athenea/lod/Lod.h` (`PairSettings::flatTolerance`), `modules/usd/src/Decimate.cpp` (writing `coverageMass`); `docs/decisions.md` ("Simplify: fewer gaussians by pairs, and surfels" → "A surfel", "Coverage").
:::

### 6.2.2 The cost

For nodes $a, b$ with $\pi = W_a/(W_a+W_b)$ and merged $\Sigma$, the shape term is Runnalls' KL bound per unit mass.
The disc's area replaces the volume, which a surfel does not have:

$$c_{\text{shape}} = \max\!\Big(0,\ \tfrac12\big[\log I_2(\Sigma) - \pi\log I_2(\Sigma_a) - (1-\pi)\log I_2(\Sigma_b)\big]\Big),$$

where $I_2$ is the second invariant (sum of pairwise eigenvalue products, computed without eigenvalues). The full cost
is

::: {.nota}
**KL divergence and Runnalls' bound** — Reducing a Gaussian mixture by merging two components $a,b$ into their moment-matched Gaussian $m$ loses information; Runnalls ([34]) gives an upper bound on the Kullback–Leibler divergence between the mixture before and after:
$$B(a,b)=\tfrac12\big[(w_a+w_b)\log\det\Sigma_m-w_a\log\det\Sigma_a-w_b\log\det\Sigma_b\big].$$
Divided by the total mass it takes the form in the text, with $\pi=W_a/(W_a+W_b)$. It is always $\ge0$ because $\log\det$ is concave over positive definite matrices and $\Sigma_m\succeq\pi\Sigma_a+(1-\pi)\Sigma_b$.

**Why $I_2$ and not $\det$** — $\det\Sigma=\lambda_1\lambda_2\lambda_3$ is the square of the ellipsoid's volume (up to constants); a surfel ($\lambda_3=0$) would have $\log\det=-\infty$. The second invariant $I_2=\lambda_1\lambda_2+\lambda_2\lambda_3+\lambda_1\lambda_3$ reduces to $\lambda_1\lambda_2$ for a disc, that is, the square of its area (up to $\pi^2$). It is computed without eigenvalues as the sum of the $2\times2$ principal minors, $I_2=\tfrac12\big[(\mathrm{tr}\,\Sigma)^2-\mathrm{tr}(\Sigma^2)\big]$ (`secondInvariant`). With $I_2$ concavity is not guaranteed, hence the $\max(0,\cdot)$.

**Scale-free** — The term is a logarithm of ratios: it does not depend on absolute size. A mass-weighted cost would merge the smallest Gaussians first, which are precisely the details; this one does not.

*References:* [34] in §10; Kullback, S., Leibler, R. (1951) *On Information and Sufficiency*; code: `shaders/athenea/lod/lod_pairs.slang` (`secondInvariant`, `mergeCost`); `docs/decisions.md` ("Simplify…" → "The cost").
:::

$$c = \sigma_{\text{px}}\big(c_{\text{shape}} + w_f\,\phi^2 + w_n(1-\cos\theta_{ab})(1+F_0/\alpha)\big) + w_c\,\lVert\Delta c_{\text{sRGB}}\rVert^2 + w_m\,\Delta_{\text{mat}},$$

where $\phi$ is the thickness the merge adds over its thicker member. The term $\sigma_{\text{px}} =
\mathrm{smoothstep}(0.5, 2, \text{px})$ removes the shape terms of merges that are under half a pixel on the
conversion camera. Defaults: $w_c=20$, $w_n=1$, $w_m=4$, $w_f=4$.

::: {.nota}
**Breakdown of the cost** (`mergeCost` in `lod_pairs.slang`):
- **$\sigma_{\text{px}}$** — $\mathrm{smoothstep}(0.5,2,\text{px})$, where px is the merge's $\sqrt{\mathrm{tr}\,\Sigma}\,f/d$ as seen from the conversion camera. If the merge is under half a pixel its shape is not visible and the geometric terms do not count; from 2 px on they count in full. `smoothstep` is the cubic interpolation $3t^2-2t^3$ between the two edges.
- **$\phi$** — Added thickness: the code estimates the merge's thick-to-middle ratio as $\sqrt{\det\Sigma\cdot\mathrm{tr}\,\Sigma/I_2^2}$ (for $\lambda_3\ll\lambda_2$, $\det/I_2\approx\lambda_3$ and $I_2/\mathrm{tr}\approx\lambda_2$) and subtracts that of the thicker member. Two discs on a curved surface merge into a lens; this term penalises it.
- **Normal** — $(1-\cos\theta_{ab})(1+F_0/\alpha)$: the turn of the normal weighs more the more specular the surface, because a highlight moves with the normal in inverse proportion to the width of the lobe. $F_0$ is the specular reflectance at normal incidence (the node's "specular" channel).
- **Colour** — $w_c\lVert\Delta c_{\text{sRGB}}\rVert^2$, with the edge guard at 0.15 (§6.2.3). With $w_c=20$, the largest admissible difference contributes $20\times0.15^2=0.45$.
- **Material** — $\Delta_{\text{mat}}=\Delta\text{metallic}^2+(\Delta\sqrt\alpha)^2+\Delta\text{transmission}^2$.

**Defaults** — $w_c=20$, $w_n=1$, $w_m=4$, $w_f=4$: `colourWeight`, `normalWeight`, `materialWeight`, `flatWeight` in `PairSettings` (`modules/lod/include/athenea/lod/Lod.h`). The resulting cost is dimensionless; in §6.3 it is converted to metres by multiplying by the node's size.

*References:* [24], [34] in §10; code: `shaders/athenea/lod/lod_pairs.slang` (`mergeCost`), `modules/lod/include/athenea/lod/Lod.h` (`PairSettings`); `docs/decisions.md` ("Simplify…" → "The cost").
:::

### 6.2.3 Guards

A pair whose members differ in any of the ways below costs $\infty$ (`kNoMerge`):

| guard | rule | default |
|---|---|---|
| part | key differs (prim, joint, sheet, thin wall, Schlick metal) | always |
| glass | transmission $>0.5$ on one side only | always |
| glass with glass | only with `--merge-glass` | off |
| mirror | $\alpha<0.01$ on one side only | always |
| fold | $\cos\theta_{ab} < \cos 60^\circ$ | 60° |
| highlight | $\alpha<0.1$ and $1-\cos\theta_{ab} > \tfrac12\kappa^2\alpha$ | $\kappa=1$ |
| colour edge | $\lVert\Delta c_{\text{sRGB}}\rVert > 0.15$ | 0.15 |
| needle | plane aspect above both `maxAspect` and 1.25× the members' | 1000 (off) |
| stroke | merge wider than `maxPixels` on the camera | off |

::: {.nota}
**Guards: cost $\infty$** — A guard is not a term of the cost but a prohibition: the pair is worth `kNoMerge` ($10^{30}$) and is never chosen. Row by row:
- **part** — The key (`lodAttributeKey` in `lod_attributes.slang`) combines the source prim (`cryptoObject`), the dominant joint, and the sheet (thin glass), thin-wall and Schlick-metal flags. Two Gaussians with different keys never merge, which is why the trees are per part (§6.2.5).
- **glass** — Transmission $>0.5$ on one side only: glass with non-glass.
- **glass with glass** — Only with `--merge-glass`: the windscreen merged 10× gave an iridescent moiré, because each sheet is drawn as a reflection layer (`kSheetAlpha` in `shaders/athenea/splat/splat_project.slang`).
- **mirror** — $\alpha<0.01$ (roughness $<0.1$) on one side only.
- **fold** — Normals more than 60° apart ($\cos\theta<0.5$, `maxAngle = 60`): a crease in the bodywork.
- **highlight** — On glossy surfaces ($\alpha<0.1$), if $1-\cos\theta>\tfrac12\kappa^2\alpha$. Since $1-\cos\theta\approx\theta^2/2$, this is equivalent to $\theta>\kappa\sqrt\alpha$: the angle between the normals exceeds the width of the lobe (of the order of $\sqrt\alpha$ rad). Two such normals put the highlight in two places, and their merge would put it in one in between (measured: the chrome ball merged 5× sparkled like glitter).
- **colour edge** — Euclidean sRGB distance greater than 0.15 (`edge = 0.15`).
- **needle** — The merge's plane aspect, $\sqrt{\lambda_1/\lambda_2}$ (`aspectOf`, computed from the trace and $I_2$), above both `maxAspect` and 1.25× that of its members. Off by default (1000): at 4 the Corvette at 10× measured worse.
- **stroke** — A merge wider than `maxPixels` px on the camera: it cannot hold the edge its members drew.

*References:* code: `shaders/athenea/lod/lod_pairs.slang` (`mergeCost`, `kNoMerge`, `aspectOf`), `shaders/athenea/lod/lod_attributes.slang` (`lodAttributeKey`), `modules/lod/include/athenea/lod/Lod.h` (`PairSettings`: `edge`, `maxAngle`, `highlightCone`, `maxAspect`), `apps/athenea/src/CmdDecimate.cpp` (`--merge-glass`, `--max-angle`, `--highlight-cone`, `--max-aspect`, `--max-pixels`); `docs/decisions.md` ("Simplify…", "What no camera sees, and merges no wider than the screen holds (9 Oct)").
:::

### 6.2.4 Passes and the relaxed top

In each pass, every node computes the cost of merging with the 8 neighbours on either side along a Morton order of
the centres. It then repeats this along a second order shifted by half the finest cell, and keeps the cheaper partner.
Pairs that chose each other are binned into a $\log_{10}$ cost histogram, and the host picks the cost threshold that
merges as many pairs as the pass may, which is at most half the nodes. Merging stops at `treeRatio` (4096×) or when
nothing more may merge. With `relaxTop`, merging then continues with the colour edge (2.0), the angle, the
glass-among-itself guard, the aspect and the pixel limit lifted. The key, glass-with-non-glass and mirror guards stay.
Each such merge records its cost **negated**, which marks it as past the guards. This gives a far view coarse levels
to cut to: one headlight cover goes from 76 486 roots to 123.

::: {.nota}
**Candidates along a Morton order** — Finding the cheapest neighbour among all nodes is $O(n^2)$. Instead, each pass computes the Morton code of the current centres (`pairCodes`), sorts, and each node evaluates the 8 neighbours on either side in that order (`kWindow = 8`, `pairBest`). The Z-order separates nearby points that fall on either side of a large cell boundary; a second order with its origin shifted by half the finest cell brings them back together. In total, up to 32 candidates per node.

**Threshold by histogram** — The mutual pairs are binned into 256 bins of $\log_{10}$ of the cost (`pairHistogram`). The host reads the histogram (counters only) and picks the cost limit that lets as many pairs merge as the pass allows, cheapest first. Each pair removes one node, so a pass at most halves the count.

**`treeRatio` (4096×)** — Merging stops when $n/4096$ nodes remain or none can merge any more (all remaining pairs are worth `kNoMerge`).

**The relaxed top (`relaxTop`, on by default)** — When the guards prevent further merging, the tree is left with thousands of roots per part, and a far view has no coarse levels to choose from. With `relaxTop` merging continues with the colour edge at 2.0 (more than the largest possible sRGB distance, $\sqrt3\approx1.73$: the guard is void), no angle limit (`maxCos = -1`), glass with glass, no aspect and no pixel limit. The key, glass/non-glass and mirror guards remain. Those merges store their cost **negated** (`treeCost[node] = -cost` in `pairMerge`), and the cut treats them separately (§6.3, factor $g$ and floor $2r$). Result: the headlight cover goes from 76 486 roots to 123.

*References:* [9], [32] in §10; code: `shaders/athenea/lod/lod_pairs.slang` (`pairCodes`, `pairBest`, `pairHistogram`, `pairMerge`), `modules/lod/src/Lod.cpp` (the pass loop, `relaxed`, `kHistogramLo`), `modules/lod/include/athenea/lod/Lod.h` (`relaxTop`, `passes`, `treeRatio`).
:::

**Layout.** The tree is written as a ParticleField. The leaves come first, in depth-first order, so that every node is
a contiguous run of leaves. The merges follow in the order they were made. Each node carries `lodParent` (a root is
its own parent), `lodStart` (its first leaf), `lodSize` (its number of leaves) and `lodCost`.

::: {.nota}
**ParticleField** — The OpenUSD schema for Gaussian particle clouds ([43]); athenea writes the tree as a single ParticleField with additional primvars.

**Depth-first order** — The leaves are placed in the order of a depth-first traversal of the tree (`pairTreeLayout`: the roots one after another, by prefix sum of their sizes; the first child of each merge at the parent's start, the second behind it). The consequence is that **every node is a contiguous run of leaves**, $[\texttt{lodStart},\texttt{lodStart}+\texttt{lodSize})$. This is the nested-interval representation of a tree (as in a *nested set*), and it lets any kernel walk a node's leaves without pointers: attribute merging (§6.2.1), the coverage fit (§6.4) and the page key (§6.8).

**The four primvars** — `lodParent` (the parent's index in the file; a root points to itself), `lodStart`, `lodSize` (1 for a leaf) and `lodCost` (0 for a leaf, negative for a relaxed-top merge).

*References:* [43] in §10; code: `shaders/athenea/lod/lod_pairs.slang` (`pairTreeOffsets`, `pairTreeSizes`), `shaders/athenea/lod/lod_pairs_final.slang` (`pairTreeLayout`, `outParent`, `outRun`, `outCost`), `modules/usd/schemas/generatedSchema.usda` (`lodCost`).
:::

### 6.2.5 Many parts: `--passes`, `--merge-glass`, `--object-groups`

On the whole Corvette (5.38 M leaves), the tree can be built in one pass, but writing its attributes exceeds device
memory. `--object-groups K --object-group i` keeps the Gaussians whose `cryptoObject` mod $K$ is $i$. Merges never
cross parts, so each group's tree is exactly the subtree the whole asset would hold for those parts. Each pass merges
at most half of what is left, and a group of many small parts merges few pairs at once. The default cap of 60 passes
leaves a group with 116 549 roots out of 323 613 leaves. At `--passes 1000` it reaches 1 298 roots in 310 passes.
`--tree-report` shows two causes of the remaining roots: solid glass never merges (roots equal leaves), and the tyres
merge a trickle per pass until the cap.

::: {.nota}
**`--object-groups K --object-group i`** — Partitions the Gaussians by `cryptoObject mod K` (the Cryptomatte part identifier, [47]) and builds the tree of one group. Because the key guard forbids merges between parts, a part's tree does not depend on which other parts are in the same process: the $K$ trees together are exactly the whole asset's tree, split. This lets the work and the memory be distributed.

**Why so many passes are needed** — Each pass's threshold is global: the cheapest pairs of the whole group merge first. A large, uniform part soon runs out of cheap pairs; many small parts each contribute few pairs per pass. With the cap of 60 passes 116 549 roots remain out of 323 613 leaves (36 %); with `--passes 1000`, 1 298 roots after 310 passes.

**`--tree-report`** — Prints the remaining roots per part (names from the cryptomatte manifest). Two causes: solid glass, which does not merge without `--merge-glass` (roots = leaves), and the tyres, which merge a few pairs each pass until the cap runs out (the text records the trickle without attributing it to a specific guard; strongly curved geometry with relief, like a tyre's, is what brings the most pairs close to the fold and edge guards).

*References:* [47] in §10; code: `modules/usd/src/Decimate.cpp` (filter by `cryptoObject`, `--tree-report`), `apps/athenea/src/CmdDecimate.cpp` (`--object-groups`, `--object-group`, `--passes`, `--merge-glass`, `--tree-report`), `modules/lod/include/athenea/lod/Lod.h` (`passes = 60`).
:::

| group (640 px car) | roots before | with `--merge-glass --passes 4000` |
|---|---|---|
| group 2 | 190 490 | 4 237 |
| group 5 | 36 508 | 817 |
| whole car (8 groups) | — | 17 133 roots, 5.38 M leaves, 3 582 pages |

::: {.nota}
**Reading the table** — "640 px car" is the bake at 640 px width of the conversion camera (5.38 M leaves). With `--merge-glass --passes 4000`, group 2 goes from 190 490 roots to 4 237 (45× fewer) and group 5 from 36 508 to 817. The whole car in 8 groups: 17 133 roots.

**Checking the figures** — In a binary tree each merge reduces the number of roots by one, so $\text{merges}=\text{leaves}-\text{roots}$ and $\text{nodes}=2\cdot\text{leaves}-\text{roots}\approx2\cdot5.38\,\text{M}-17\,133\approx10.74$ M. With pages of 3000 nodes (§6.8), $10.74\,\text{M}/3000\approx3\,582$: the table's 3 582 pages.

*References:* code: `apps/athenea/src/CmdDecimate.cpp` (`--tree-pages`).
:::

## 6.3 The cut by error, with bounds

**Alternative considered: the size cut.** `--cut-pixels` keeps a node when its on-screen width $2\max(s)f/d$ is at
most $\tau$ and no ancestor's is. Merges past the guards pass only at $0.25\tau$. That cut is still available. The
error cut below, after Nanite, gives higher PSNR with fewer nodes (table at the end of this section).

::: {.nota}
**Size cut** — The analogue of the octree cut (§6.1) on the merge tree: a node is kept if its on-screen width, $2\max(s)\,f/d$ (the $1\sigma$ diameter of the major axis), is at most $\tau$ and no ancestor satisfies the same (`treeCutWidth`, `treeCutKeep`). Relaxed-top merges pass only at $0.25\tau$ (`relaxedShare = 0.25` in `Decimate.cpp`). Its flaw: it measures how much a merge occupies, not how wrong it is. A large merge of a uniform part could be drawn much earlier than a small one that blends an edge.

**Cut by error, after Nanite** — In Nanite ([35]) each cluster group stores the geometric error of its simplification and a sphere; a cluster is drawn if its projected error is $\le\tau$ and its parent's is $>\tau$. The same logic appears in the *screen-space error* of OGC/Cesium 3D Tiles ($\text{SSE}=\text{geometricError}\cdot H/(2d\tan(\mathrm{fov}/2))$, that is, $\varepsilon f/d$). The header comment of that section in `lod_tree_cut.slang` cites both sources.

*References:* [35] in §10; OGC (2019) *3D Tiles Specification 1.0* (`geometricError`); code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCutWidth`, `treeCutKeep`), `modules/usd/src/Decimate.cpp` (`relaxedShare`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-pixels`).
:::

**Own error.** Let $r_n=\lVert s_n\rVert$ (the square root of $\mathrm{tr}\,\Sigma_n$) and $\alpha_n =
\max(\text{roughness},0.02)^2$. A merge's own error in metres is

$$\varepsilon_0(n) = r_n\Big(|\mathrm{cost}_n| + w_c\,s_c(n) + w_n\,\frac{\theta(n)^2}{\alpha_n} + w_s\max(E_n-E_0,0)\Big),\qquad w_c=10,\ w_n=1,$$

and $\varepsilon_0=0$ for a leaf. Merges past the guards ($\mathrm{cost}_n<0$) are not measured by their cost, which
misses the moiré it causes on glass and the colour it drags across an edge. They are therefore held to

$$\varepsilon_0(n) \leftarrow \max\big(g\,\varepsilon_0(n),\ 2r_n\big),\qquad g=4\ (\texttt{--cut-guard}).$$

The spill term $w_s$ is 0 by default (§6.5).

::: {.nota}
**Own error $\varepsilon_0$, in metres** — The merge cost (§6.2.2) is dimensionless; multiplying it by $r_n=\lVert s_n\rVert=\sqrt{\mathrm{tr}\,\Sigma_n}$ (the node's "size", the square root of the sum of its variances) turns it into a length that can be projected to pixels. The terms:
- $|\mathrm{cost}_n|$: the cost of the last merge (in absolute value, because of the relaxed ones).
- $w_c\,s_c(n)$: the bounded colour spread (next paragraph), $w_c=10$.
- $w_n\,\theta(n)^2/\alpha_n$: the squared normal cone, relative to the lobe width; $\alpha_n=\max(\text{roughness},0.02)^2$ avoids dividing by zero on a perfect mirror. $w_n=1$.
- $w_s\max(E_n-E_0,0)$: the spill, disabled ($w_s=0$, §6.5).

**Relaxed merges** — Their cost was computed with the guards lifted and does not see what they break (the moiré of merged glass, colour dragged across an edge). Their error is therefore multiplied by $g=4$ (`--cut-guard`) and never falls below $2r_n$. With the floor $2r_n$, $P\ge2r_nf/d$: a relaxed merge enters a cut at $\tau$ only when the node measures at most $\tau/2$ on screen (in terms of $r_n$).

**Implementation** — `treeErrorBounded` in `lod_tree_cut.slang`: `e = size * (|cost| + colourWeight·spread + normalWeight·θ²/α + spillWeight·max(spill − spillFloor, 0))`, and if `cost < 0`, `e = max(guard·e, floorSize·size)`. `Decimate.cpp` sets `colourWeight = 10`, `normalWeight = 1`, `floorSize = 2` and takes `guard` from `--cut-guard`. A leaf has error 0 (`treeErrorOwn`).

*References:* code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorOwn`, `treeErrorBounded`), `modules/usd/src/Decimate.cpp` (cut parameters), `apps/athenea/src/CmdDecimate.cpp` (`--cut-error`, `--cut-guard`, `--cut-bounds`, `--cut-spill`, `--cut-spill-floor`).
:::

**Bounds.** A merge's cost covers only the last pair it joined. A chain of merges, each within the colour edge, can
drift across that edge. So a colour spread $s_c$ and a normal cone $\theta$ are climbed up the tree by the triangle
inequality (`treeBoundsUp`), starting at 0 for the leaves:

$$s_c(p) = \max_{k\in\mathrm{ch}(p)}\big(s_c(k)+\lVert c_k-c_p\rVert\big),\qquad \theta(p) = \max_{k}\big(\theta(k)+\angle(n_k,n_p)\big).$$

Both are therefore bounds on every leaf under $p$.

::: {.nota}
**Why bounds are needed** — A merge's cost compares only the two children it joins. A chain A–B–C–D in which each step changes the colour by 0.1 respects the 0.15 edge at every merge, but the final node mixes A and D, which differ by 0.3. This is the drift of agglomerative clusterings with a local criterion (the "chaining effect" of *single linkage*).

**The triangle inequality** — If $s_c(k)$ bounds the colour distance of every leaf $l$ under $k$ to the colour of $k$, then for the parent $p$:
$$\lVert c_l-c_p\rVert\le\lVert c_l-c_k\rVert+\lVert c_k-c_p\rVert\le s_c(k)+\lVert c_k-c_p\rVert,$$
and taking the maximum over the children gives $s_c(p)$. By induction from the leaves ($s_c=0$), $s_c(p)$ bounds every leaf's deviation. The same holds for the normals with the angular distance $\angle(n_k,n_p)$, which is also a metric on the sphere: $\theta(p)$ is the half-angle of a cone around $n_p$ that contains every leaf's normal. It is the idea of the *normal cones* of hierarchical back-face culling (Shirman and Abi-Ezzi, 1993).

**Parallel computation** — `treeBoundsUp` launches one thread per node; each raises its parent's value with `InterlockedMax` on the float's bits (for non-negative floats, IEEE 754 order coincides with unsigned integer order). The kernel is repeated until the `changed` counter reads 0; the number of rounds is the depth of the tree.

*References:* Shirman, L. A., Abi-Ezzi, S. S. (1993) *The Cone of Normals Technique for Fast Processing of Curved Patches*, Computer Graphics Forum (Eurographics); Johannsen, A., Carter, M. B. (1998) *Clustered Backface Culling*, Journal of Graphics Tools; code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeBoundsUp`, buffers `spreadBits`, `coneBits`), `modules/usd/src/Decimate.cpp` (the relaunch loop).
:::

**Monotone error and covering spheres.** The error is made monotone, and every node gets a sphere that contains its
children's spheres:

$$\varepsilon(n) = \max\big(\varepsilon_0(n),\ \max_{k}\varepsilon(k)\big),\qquad \rho(n)=\max_k\big(\lVert\mu_n-\mu_k\rVert+\rho(k)\big),\quad \rho(\text{leaf})=2\max(s).$$

Each climb is one kernel launched over all nodes, in which every node raises its parent with `InterlockedMax` on the
float's bits. The kernel is relaunched until a counter reads zero, which is the tree's depth in rounds. The error is
projected at the sphere's nearest point:

$$P(n) = \frac{\varepsilon(n)\,f}{\max(\lVert\mu_n-e\rVert-\rho(n),\ 10^{-3})}.$$

::: {.nota}
**Monotone error** — $\varepsilon(n)=\max(\varepsilon_0(n),\max_k\varepsilon(k))$: a node's error is raised to the maximum of its descendants'. Without this, a cheap merge above an expensive merge would have less error than its child, and the selection could draw the parent (below $\tau$) together with the child (above $\tau$ but with a "good" parent). Nanite imposes the same condition ([35]).

**Covering spheres** — $\rho(\text{leaf})=2\max(s)$ (the $2\sigma$ envelope of the major axis) and $\rho(p)=\max_k(\lVert\mu_p-\mu_k\rVert+\rho(k))$: by the triangle inequality, the ball with centre $\mu_p$ and radius $\rho(p)$ contains each child's, and by induction all leaves under $p$.

**Projected error** — $P(n)=\varepsilon(n)f/\max(d-\rho,10^{-3})$ with $d=\lVert\mu_n-e\rVert$: the error divided by the distance to the sphere's **nearest point** (the most pessimistic estimate for any part of the node). If the eye is inside the sphere, the distance is clamped to 1 mm and $P$ becomes huge: the node is refined.

**The climbs** — The same scheme as the bounds: one kernel per round with `InterlockedMax` on the bits (`treeErrorUp` climbs error and radius; `treeErrorClimb` only the error, once the spheres are fixed), relaunched until nothing changes.

*References:* [35] in §10; code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorUp`, `treeErrorClimb`, `projectedError`, buffers `errorBits`, `radiusBits`), `shaders/athenea/lod/lod_pages.slang` (`projectedError`, the same formula).
:::

**Proposition.** If $P(n)\le\tau<P(\mathrm{par}(n))$ selects $n$ (a root needs only the first inequality), then every
leaf is drawn exactly once. *Proof.* $\varepsilon$ does not decrease going up. The parent's sphere contains the
child's, so the distance to the sphere decreases going up. $P$ therefore does not decrease along any leaf-to-root
path. Leaves have $P=0\le\tau$, so the predicate "$P\le\tau$" holds on a prefix of the path starting at the leaf, and
exactly one node is the last of that prefix. $\square$

::: {.nota}
**The proposition, step by step** — Let $l=n_0,n_1,\ldots,n_m$ (the root) be the path from a leaf.
1. *$\varepsilon$ does not decrease*: by construction, $\varepsilon(n_{i+1})\ge\varepsilon(n_i)$.
2. *The distance to the sphere does not grow*: $\lVert\mu_{i+1}-e\rVert\le\lVert\mu_i-e\rVert+\lVert\mu_{i+1}-\mu_i\rVert$ and $\rho_{i+1}\ge\rho_i+\lVert\mu_{i+1}-\mu_i\rVert$; subtracting, $d_{i+1}-\rho_{i+1}\le d_i-\rho_i$. The $\max(\cdot,10^{-3})$ is monotone, so it preserves the inequality.
3. By 1 and 2, $P(n_{i+1})\ge P(n_i)$.
4. $P(l)=0\le\tau$. The set $\{i: P(n_i)\le\tau\}$ is a prefix $\{0,\ldots,j\}$ of the path; $n_j$ satisfies $P(n_j)\le\tau<P(n_{j+1})$ (or is the root). No other node of the path satisfies it: below $j$ the parent is also under $\tau$; above, the node itself is not.

Since the cut is the union of those nodes over all leaves, every leaf has exactly one representative: no holes and no double draws.

**Comparison with §6.1** — It is the same argument as the monotonicity of the octree, but the error is no longer a cell's edge: it is a quantity of the content (cost, colour, normals) made monotone on purpose.

*References:* [35] in §10; code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorKeep`).
:::

So the cut is a single parallel test per node (`treeErrorKeep`), with no walk up the ancestors. `--cut-budget N`
bisects $\log\tau$ over $[10^{-4},10^4]$ in 40 steps for the smallest threshold whose cut holds at most $N$ nodes.

::: {.nota}
**One test per node** — `treeErrorKeep` evaluates in each thread `P(i) <= threshold && (root || P(parent) > threshold)`: there is no need to climb the ancestors or walk the tree, and the cost is $O(\text{nodes})$ per frame with full parallelism.

**`--cut-budget N` and bisection** — The number of nodes in the cut is non-increasing in $\tau$ (as $\tau$ rises each leaf comes to be represented by an equal or higher ancestor). The smallest $\tau$ that meets the budget can therefore be found by bisection. The bisection is on $\log\tau$ because the range spans 8 orders of magnitude ($10^{-4}$ to $10^4$ px); 40 steps shrink the logarithmic interval by a factor of $2^{40}\approx10^{12}$, far more than needed, but each step is just one dispatch and one counter read.

*References:* code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorKeep`), `modules/usd/src/Decimate.cpp` (bisection: `low = log(1e-4)`, `high = log(1e4)`, 40 iterations), `apps/athenea/src/CmdDecimate.cpp` (`--cut-budget`).
:::

| configuration | scene | size cut | error cut |
|---|---|---|---|
| far, 393 px, 0.25 px | six pieces | 83 k, 34.4 dB | 34 k, 34.9 dB |
| near, 426 k nodes | six pieces | 42.92 / 31.22 dB (frame / grille) | 43.61 / 34.50 dB (reference prototype 43.61 / 34.47) |

::: {.nota}
**Reading the table** — "Six pieces" is a test scene with six parts of the Corvette (including the grille and the headlight cover). Row 1: far view rendered at 393 px width (phone size) with $\tau=0.25$ px: the size cut needs 83 k nodes for 34.4 dB and the error cut 34 k (2.4× fewer) for 34.9 dB. Row 2: close view with both cuts set to the same budget of 426 k nodes; PSNR of the full frame and of the grille crop. The error cut gains 0.7 dB on the frame and 3.3 dB on the grille, the hard region. The "reference prototype" is the earlier research implementation (43.61/34.47): it agrees with the kernel to 0.03 dB.

**PSNR** — $10\log_{10}(\text{MAX}^2/\text{MSE})$; +3 dB amounts to halving the MSE. The full-frame PSNR includes the background (sky, ground) where there is no error, and dilutes the differences; that of a crop measures the region that matters. That is why the text gives both.

*References:* code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCutKeep`, `treeErrorKeep`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-pixels`, `--cut-budget`).
:::

## 6.4 The opacity of a merge: least-squares coverage

Mass over the moment ellipse counts the leaves' overlapping tails twice. The overcount measures 27 % at the median and
2.3× at p10. On the grille at 393 px, merges draw darker than their leaves. `--cut-coverage` (at cut time) and
`--tree-coverage` (when the tree is written) set every merge's opacity from what its leaves actually cover, as
follows.

1. The node's plane is spanned by its two widest axes $u,v$, with scales $s_u,s_v$. An $8\times8$ grid covers the
   core $[-1.25,1.25]^2$ in units of $(s_u,s_v)$.
2. Each leaf $l$ under the node (its run, up to 4096 leaves when writing the tree and 2048 in a cut) is projected onto
   the plane. Its in-plane covariance is $(u^\top\Sigma_lu,\ u^\top\Sigma_lv,\ v^\top\Sigma_lv)$ and its centre
   offset is $m_l$. A longer run keeps its mass-based opacity. The leaves are composited as the rasteriser would:
   $C(x)=1-\prod_l\big(1-\min(o_l,0.99)\,G_l(x)\big)$.
3. The node's own footprint on the grid is $G(x)=\exp(-\tfrac12\lVert x\rVert^2)$. The opacity is the least-squares
   fit of $aG$ to $C$:

$$a^\star = \frac{\sum_x G(x)\,C(x)}{\sum_x G(x)^2},\qquad o_n = \mathrm{clamp}(a^\star,0,0.99).$$

::: {.nota}
**Why the mass overestimates** — $W=\sum o_lA_l$ adds the leaves' areas as if they did not overlap. But the Gaussians of a bake overlap their neighbours (their tails reach beyond the spacing), and the real coverage of a stack of semi-transparent layers is not the sum but $1-\prod(1-\alpha_l)$. The mass counts the overlap twice: an overcount of 27 % at the median and 2.3× at the 10th percentile. On the grille at 393 px, the measured result is that the merges come out darker than their leaves: they cover more than the leaves did, and hide more of what lies behind.

**The procedure** (`treeCoverage` in `lod_tree_cut.slang`):
1. The node's plane is spanned by its two widest axes; it is sampled with $8\times8$ points over $[-1.25,1.25]^2$ in units of its scales.
2. Each leaf is projected onto the plane: its 2D covariance is the restriction $(u^\top\Sigma_lu, u^\top\Sigma_lv, v^\top\Sigma_lv)$ and its centre $m_l$. It is composited as the rasteriser does, $C(x)=1-\prod_l(1-\min(o_l,0.99)G_l(x))$: the accumulated opacity of "over" *alpha blending* (Porter and Duff, 1984), which does not depend on the order.
3. The amplitude $a$ of the node's own Gaussian, $aG(x)$, is fitted to $C(x)$ by **least squares**: minimising $\sum_x(aG-C)^2$, differentiating with respect to $a$ and setting to zero gives $a^\star=\sum GC/\sum G^2$. If $C$ were exactly $kG$, $a^\star=k$: the estimator recovers the amplitude.

Runs of more than 4096 leaves (when writing the tree) or 2048 (in a cut) keep the mass-based opacity: they are coarse nodes of far views, where it shows least, and the cost of the fit grows with the run.

*References:* Porter, T., Duff, T. (1984) *Compositing Digital Images*, SIGGRAPH; code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCoverage`, `covarianceOf`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-coverage`, `--tree-coverage`).
:::

Two alternatives were measured. Normalising by $\sum G$ instead of $\sum G^2$ gives 0.5 for a merge as wide as one
opaque leaf. A fit over $\pm2\sigma$ instead of $\pm1.25\sigma$ gives a panel's merges 0.84, because the fringe is
where the neighbours carry the surface, and a fender comes out mottled. Leaves edge-on to the plane, such as a
grille's wires seen square on, cover nothing, so the merge goes to 0. A loader drops anything under 1/255, which would
break the tree, so a merge is written at least at

$$o_n \ge 2/255\quad(\texttt{treeCoverageFloor}).$$

::: {.nota}
**Normalising by $\sum G$** — $\sum GC/\sum G$ is a $G$-weighted mean of $C$, not an amplitude: even if $C$ has exactly the shape of $G$, it gives a value below its peak (the mean of a bell is less than its maximum). Hence the 0.5 measured for a merge as wide as one opaque leaf.

**Window $\pm1.25\sigma$ against $\pm2\sigma$** — Outside the core, what covers the surface are the node's neighbours, not its own leaves: a fit over $\pm2\sigma$ sees half-empty tails and lowers the opacity (0.84 for a panel's merges), and the fender comes out mottled because the background shows through the centre of each merge.

**Edge-on leaves** — A leaf whose plane contains the merge's normal direction (a grille's wires seen square on) projects to a degenerate ellipse: it covers nothing on the plane, $C\approx0$ and the opacity drops to 0. That is correct for the coverage on that plane, but it breaks loading: the loader's validator drops every record with opacity below 1/255 (`splat_validate.slang`), and a dropped merge would orphan its children. Hence the $2/255$ floor (`kTreeFloor` in `treeCoverageToFile`; `treeCoverageFloor` restores the floor on merges the loader had already dropped).

*References:* code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCoverage`, `kTreeFloor`, `treeCoverageToFile`, `treeCoverageFloor`), `shaders/athenea/scene/splat_validate.slang`.
:::

The tree is rewritten in its own file order, and only the merges' opacities change (`treeCoverageToFile`), because
`lodParent`/`lodStart` are file indices that a rewrite through the Morton-ordered store would scramble. The fit reads a
Blob's axes as rows. Read as columns, they give the inverse rotation and turn every Gaussian the wrong way: a fender's
merges then come out at 0.59 of their mass where a reference fit gives 1.00.

::: {.nota}
**Rewriting in file order** — On loading, `LodBuilder` reorders the store by Morton code (`storeOrder` maps store to file). The `lodParent`/`lodStart` indices point to file positions; a rewrite from the reordered store would invalidate them. That is why `treeCoverageToFile` writes an array of opacities indexed by file position (NaN on the leaves = leave untouched) and only the merges' opacities change.

**Rows against columns** — In `Blob` (`lod_blob.slang`) the Gaussian's axes are the **rows** of `axes`. A rotation matrix is orthogonal: its transpose is its inverse. Reading the axes as columns applies the inverse rotation, and every Gaussian ends up turned. With that bug the fit gave 0.59 of the mass on a fender's merges, where the reference gave 1.00. `covarianceOf` carries a comment about this bug.

*References:* code: `shaders/athenea/lod/lod_blob.slang` (`Blob`, `blobOf`), `shaders/athenea/lod/lod_tree_cut.slang` (`covarianceOf`, `treeStoreOf`, `treeCoverageToFile`).
:::

| configuration | what | coverage | mass |
|---|---|---|---|
| fit at cut time (48 k) | grille, 393 px | 39.79 dB (from 33.87) | — ; leaves 41.59 |
| refitted tree (49.4 k) | grille, 393 px | 41.28 dB | 39.79 at 48.1 k (`--cut-coverage`) |
| 1 px, bounds, axes read as columns | six pieces near at 1920 | 40.96 frame / 32.09 grille | old tree 41.11 / 30.09 |
| axes read as rows | six pieces near at 1920 | **43.86** | 42.55 |
| axes read as rows | six pieces far at 393 | 39.95 | 39.66 |
| axes read as rows | grille near at 393 | 39.71 | **41.16** |

::: {.nota}
**Reading the table** — Columns: PSNR with coverage opacity and with mass opacity.
- *Fit at cut time (48 k)*: on the grille at 393 px coverage rises from 33.87 to 39.79 dB; drawing all leaves gives 41.59 (the reference ceiling).
- *Refitted tree (49.4 k)*: writing the coverage into the tree (`--tree-coverage`) gives 41.28 dB, against 39.79 with the fit at cut time (which fits only the nodes in the cut).
- *Axes read as columns*: with the bug of §6.4, the new tree (40.96 frame / 32.09 grille) barely improved on the old one (41.11 / 30.09).
- *Axes read as rows*: once fixed, coverage wins on the near frame (43.86 against 42.55) and on the far one (39.95 against 39.66), but loses on the near grille (39.71 against 41.16). Bold: the better of each row.

*References:* code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCoverage`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-coverage`, `--tree-coverage`).
:::

The grille is the exception. Neighbouring merges overlap and composite as $1-\prod$, so a little extra opacity darkens
the whole patch. The web player (Spark) draws each node's mass ($\min(\text{coverageMass},0.99)$), and the engine's
rasteriser keeps the coverage.

::: {.nota}
**The grille exception** — On a continuous surface, neighbouring merges overlap little and each one's coverage is the surface's coverage. On a grille, the merges of parallel strips overlap one another, and the composition $1-\prod(1-\alpha)$ is very sensitive to opacity when there are many layers: raising each $\alpha$ a little darkens the whole patch. The mass opacity, lower in that case, lets more light through between strips.

**Two readers, two opacities** — The web player (Spark, [11]) draws $\min(\text{coverageMass},0.99)$; the engine's rasteriser uses the coverage-fitted opacity. The file carries both.

*References:* [11] in §10; code: `modules/usd/src/Decimate.cpp` (`coverageMass`).
:::

## 6.5 The spill metric, and why it failed for the grille

At a coarse cut the grille's holes close: in the 0.25 px cut of the 640 px car at 1920 px with back-face culling, the
near-ground region reads 2.24× the path trace, against 1.18 in the leaves. The cause is two-leaf merges of the
sub-pixel strips of `Chrome_Part_Grill`, seen face on and 3–6 px wide at $\tau=0.25$. Their cost is ~0 because a
pair's coverage, measured on its own plane, does not change. A spill is defined on the coverage fit's own grid:

$$E_n = \frac{\sum_x \max\big(a G(x) - C(x),\,0\big)}{\sum_x aG(x)}\in[0,1],$$

that is, the share of what the merge paints that its leaves do not. It matches a NumPy reference implementation to a
correlation of 0.994, and is written as `lodSpill`. The cut adds $r\,w_s\max(E-E_0,0)$ to the own error.

::: {.nota}
**The symptom** — In a coarse cut the grille's holes close: the region of ground seen through it reads 2.24× the path trace value (against 1.18× with all leaves), that is, a surface appears where the ground should be visible.

**The cause** — `Chrome_Part_Grill` is made of sub-pixel strips. Two adjacent leaves of a strip merge at ~0 cost: measured on its own plane, the pair's coverage does not change. At $\tau=0.25$ and face on, those merges are 3–6 px wide on screen.

**The spill $E_n$** — On the same grid as the fit of §6.4, the share of what the merge paints ($aG$) that its leaves do not paint ($C$): $\sum\max(aG-C,0)/\sum aG\in[0,1]$. It is 0 if the merge stays under the composite of its leaves at every point and 1 if it paints where no leaf paints. The implementation agrees with a NumPy reference with a correlation of 0.994. It is written as `primvars:athenea:splat:lodSpill`, and the cut can add it to the own error with weight $w_s$ and floor $E_0$ (`--cut-spill`, `--cut-spill-floor`).

*References:* code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCoverage`, buffer `spill`; `treeErrorBounded`, `spillWeight`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-spill`, `--cut-spill-floor`).
:::

| 640 px car, group 0, 0.25 px | nodes | car crop | frame |
|---|---|---|---|
| $w_s=0$ | 307 665 | 16.55 dB | 22.18 dB |
| $w_s=2$ | 561 242 (+82 %) | 16.79 | 22.38 |
| $w_s=5$ | 599 983 | 16.79 | 22.38 |
| $w_s=10$ | 612 963 (+99 %) | 16.79 | 22.38 |

::: {.nota}
**Reading the table** — Group 0 of the 640 px car, cut at 0.25 px. At $w_s=2$ the node count rises by 82 % (307 665 → 561 242) to gain 0.24 dB on the car crop and 0.20 dB on the frame. Raising $w_s$ to 5 or 10 adds still more nodes (up to +99 %) without improving anything: the term has already saturated what it can fix.

*References:* code: `apps/athenea/src/CmdDecimate.cpp` (`--cut-spill`).
:::

The spill closes the holes: from 1.90 to 1.09 of the leaves in the render, and from 4.66× to 1.0 in an image-space
hole proxy. The price is 82–99 % more nodes. Every merge spills about 0.16, and the merges that fill the holes spill
no more than a panel's merges do, so no floor $E_0$ separates the two groups. The hole is a property of the
**projection**: the merge covers nothing in 3D that its neighbours do not cover, and the gap opens only where the
strips' screen footprints fall apart. No view-independent per-node measure sees that. The default is $w_s=0$.
The 1280 px bake (§6.10) resolves the problem in the data instead: the grille's holes in the cut go from 2.24 to 1.04.

::: {.nota}
**Why no per-node measure sees it** — The quantification: the spill closes the holes (from 1.90× to 1.09× of the leaves in the render; from 4.66× to 1.0 in an image-space hole proxy), but every merge spills about 0.16 and the ones that fill holes spill no more than a panel's. A floor $E_0$ cannot separate the two groups.

**A property of the projection** — In 3D, the merge of two pieces of a strip covers nothing that the strip and its neighbours do not cover: the gap between strips is not "inside" any node. The gap appears only on screen, where the projected footprints of sub-pixel strips are separated by empty pixels; a merge several pixels wide spreads over that gap *in the image*. Whether the gap is visible depends on the view (direction, distance, and the rasteriser's screen filter, which widens every sub-pixel Gaussian; see Mip-Splatting, [4]). A view-independent measure cannot capture it.

**The fix in the data** — With the 1280 px bake (§6.9–6.10), the grille's leaves are finer and the cut resolves the strips better: the hole proxy goes from 2.24 to 1.04 without spill. That is why $w_s=0$ by default.

*References:* [4] in §10.
:::

## 6.6 Back-face culling of closed shells

At small output sizes, a cut shows back faces as black. On the grille at 393 px, culling back faces raises the uncut
leaves from 38.2 to 41.6 dB. `--cut-cull-back` drops a node whose whole normal cone faces away:

$$\theta(n)<\tfrac{\pi}{2}\ \wedge\ n_n\cdot\frac{e-\mu_n}{\lVert e-\mu_n\rVert} < -\sin\theta(n).$$

::: {.nota}
**Back faces in a cut** — In a closed mesh, the faces looking away are hidden by the front ones. With Gaussians at sub-pixel scale, the front face does not always cover the pixel fully, and the back face (lit by the dark interior of the part) adds black. Culling the back faces raises the grille's leaves at 393 px from 38.2 to 41.6 dB.

**The cone condition** — Let $v=(e-\mu_n)/\lVert e-\mu_n\rVert$ be the direction to the eye. A normal faces away if $n\cdot v<0$. All normals within a cone of half-angle $\theta$ around $n_n$ face away if the angle between $n_n$ and $v$ exceeds $\pi/2+\theta$, that is, if $n_n\cdot v<\cos(\pi/2+\theta)=-\sin\theta$. The condition $\theta<\pi/2$ excludes cones spanning more than a hemisphere, which can never face away entirely. It is the *cone of normals* test for hierarchical back-face culling (Shirman and Abi-Ezzi, 1993; Johannsen and Carter, 1998, *clustered backface culling*), the same one used by today's GPU *meshlets*.

*References:* code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorKeep`, the block "ITS BACK, NOT DRAWN"; `treeCulledMark`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-cull-back`).
:::

The cone bounds every leaf under the node (§6.3), so the node's whole subtree faces away too. Two measurements set
where it applies. Applied to the six test pieces with the headlight's glass cover among them, it costs 6 dB at 1920:
glass is seen through, so its back faces are visible. On the whole car at 393 px with the 1 px error cut, the doors
and panels, which are two skins, show a net of dark cracks: the inner skin's merges, facing away and lit by the dark
interior, sort in front of the outer skin in patches. The net is the same with mass or coverage opacity, and
`--cut-guard 1000` does not change it. On a closed shell, what lies behind a face that looks away is the part's own
inside, so culling it is safe and removes the net. The cull is therefore applied, to leaves and merges alike, only on
**closed shells that are not glass** (`primvars:athenea:splat:closed`, transmission $\le 0.5$).

::: {.nota}
**Why it is safe for the subtree** — $\theta(n)$ bounds the normal of every leaf under $n$ (§6.3), so if the cone faces away entirely, every leaf faces away too: the whole subtree can be dropped (`treeCulledMark` marks it so that a traversal stops there).

**Where not to apply it** — Glass: it is seen through, and its back faces are part of what is visible (culling on the six pieces, with the headlight cover, cost 6 dB). Two-skin surfaces (doors and panels with inner and outer sheet metal): without culling, the merges of the inner skin, facing inwards and lit by the dark interior, sort in front of the outer skin in patches and form a net of dark cracks; the net depends neither on the opacity (mass or coverage) nor on the guards (`--cut-guard 1000`).

**Closed shell** — A closed surface (with no boundary) divides space into inside and outside. If a part is closed, what lies behind a face looking away is its own inside: dropping that face cannot uncover anything that should be seen. The flag is written per Gaussian as `primvars:athenea:splat:closed`, and the cut combines it with transmission $\le0.5$ (buffer `cullable`).

*References:* code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorKeep`, `cullable`, `treeCulledMark`), `modules/usd/src/Export.cpp` (writing `primvars:athenea:splat:closed`), `modules/usd/include/athenea/usd/Export.h` (`closed`, `markClosed`), `apps/athenea/src/CmdDecimate.cpp` (`--mark-closed`).
:::

**The open-edge rule.** `mesh2splat` counts the edges not shared by exactly two triangles, $O$. Two simpler rules were
considered: $O=0$, and a share of open edges under 2 % (111 of 192 meshes closed). An open sheet's border grows as the
square root of its area, so a fine tube passes the share rule: a rim's barrel at 1 % is culled, and light shows
between the spokes. The rule used is

$$\text{closed} \iff O < 2\sqrt{E},\qquad E = 1.5\,T,$$

for a mesh of $T$ triangles. Under it, doors (0), logos (1.4) and calipers (1.5) are closed, and rims (4.0), tyres
(3.1), the hood (3.1) and the bumper (4.4) are not, which gives 102 of the Corvette's 192 meshes. With all leaves at
1920 px against the path trace, the cull lifts the whole door ~12 % evenly (door 1.01, near ground 1.02 of the path
trace), with 4.35 M → 3.70 M Gaussians and car crop 16.73 → 16.75 dB.

::: {.nota}
**Open edges** — In a closed, well-formed triangle mesh (a *2-manifold* without boundary) every edge is shared by exactly two triangles; since each triangle has 3 edges, $E=3T/2=1.5T$. Edges with 1 triangle (boundary) or more than 2 (non-manifold) count as open, $O$. `mesh2splat` counts them in a kernel (`openEdges`), welding vertices by position.

**Why $\sqrt E$** — An open sheet's border grows as the square root of its area. For an $N\times N$ grid of squares (two triangles each), $T=2N^2$, $E\approx3N^2$ and the border has $4N=4\sqrt{E/3}\approx2.31\sqrt E$ edges. The threshold $O<2\sqrt E$ therefore excludes any "square" open sheet, whatever its resolution. A share rule ($O/E<2\%$) fails on fine meshes: with many triangles, $O/E\sim1/\sqrt E$ becomes small even though the border is real, and a rim's barrel (1 %) passed as closed, with light visible between the spokes.

**The figures** — In parentheses, $O/\sqrt E$: doors 0, logos 1.4, calipers 1.5 (closed); rims 4.0, tyres 3.1, hood 3.1, bumper 4.4 (open). 102 of 192 meshes come out closed (against 111 with the 2 % rule). With all leaves at 1920 px, the cull raises the brightness of the whole door ~12 % evenly (door 1.01 and near ground 1.02 of the path trace: now almost exact), drops 0.65 M Gaussians (4.35 M → 3.70 M) and improves the car crop by 0.02 dB.

*References:* code: `apps/athenea/src/CmdMesh2Splat.cpp` (the block "A COUNTER BACK: whether the mesh is a closed shell", `closed_[h.piece] = double(open) < 2.0 * std::sqrt(meshEdges)`), `modules/usd/src/Export.cpp`.
:::

## 6.7 Occlusion masks

A TX transfer stores 256 shadow bits per Gaussian, one per cell of a $16\times16$ octahedral grid over the world's
sphere, set where a ray from the Gaussian reached the sky. A direction whose cells are all closed is one from which an
eye outside the asset cannot see the Gaussian. The shading bits cannot be reused for this: a merge holds its leaves'
bits by majority, and the player's codebook (`shadow_vq`, §7.3) is also a majority, so both would close directions from which
some leaf is seen. `--occlusion-mask N` (`occlusion_mask.slang`, `Occlusion.cpp`) builds a conservative mask per node.

::: {.nota}
**The TX transfer's shadow bits** — In §4, each baked Gaussian stores 256 bits: one per cell of a $16\times16$ octahedral grid over the sphere of world directions, set to 1 where a ray from the Gaussian in that direction escaped to the sky.

**Octahedral mapping** — Projects the sphere onto an octahedron and unfolds it into a square (Engelhardt and Dachsbacher, 2008; Cigolle et al., 2014): a nearly area-uniform parametrisation, cheap to encode (`octEncode`), with a fold at the square's edge that connects the lower hemisphere. A cell spans about $180^\circ/16\approx11^\circ$.

**From shadow to visibility** — If all cells of a direction are closed, from that direction there is an occluder between the Gaussian and infinity; an eye placed outside the asset (beyond every occluder) in that direction cannot see the Gaussian. It is precomputed, direction-dependent *occlusion culling*, in the spirit of *potentially visible sets* (Teller and Séquin, 1991) but per element and per direction.

**Why not reuse the shading bits** — A merge holds its leaves' bits by majority and so does the player's codebook (`shadow_vq`): for shading that is correct, but not for culling, because it would close directions from which some leaf is in fact visible. Culling requires **conservative** masks: opening too much is safe (something invisible is drawn), closing too much is not (something visible disappears).

*References:* Engelhardt, T., Dachsbacher, C. (2008) *Octahedron Environment Maps*, VMV; Cigolle, Z. et al. (2014) *A Survey of Efficient Representations for Independent Unit Vectors*, JCGT; Teller, S., Séquin, C. (1991) *Visibility Preprocessing for Interactive Walkthroughs*, SIGGRAPH; code: `shaders/athenea/lod/occlusion_mask.slang`, `shaders/athenea/lod/shadow_vq.slang`, `modules/usd/src/Occlusion.cpp`, `shaders/athenea/common/octahedral.slang`.
:::

**Masks.** A leaf's mask $M_l$ is its open cells dilated by one cell ($3\times3$), mirrored across the octahedral
square's edge at the fold. One cell is ~11°, and without the margin an occluder's edge closes directions from which a
grille's chrome is still visible. The mask is all open on glass (transmission $>0.5$, since what lies behind glass is
seen through it) and on a leaf with no bits. A merge's mask is the OR of its children's, climbed until no mask changes:

$$M_p = \bigvee_{k\in\mathrm{ch}(p)} M_k .$$

::: {.nota}
**$3\times3$ dilation** — Each open cell opens its 8 neighbours (`occlusionLeaf`). With cells of ~11°, an occluder's edge is quantised; without a margin, a cell marked closed may include directions from which the grille's chrome is still visible past the edge. The neighbourhood crosses the edge of the octahedral square by mirroring the cell (`across`), because at the fold the geometric neighbour is on the other side.

**All-open cases** — Glass (transmission $>0.5$: what lies behind is seen through it, even though its rays stopped on it) and leaves without bits.

**OR upwards** — $M_p=\bigvee_kM_k$: a direction is closed on a merge only if it is closed on all its children (and, by induction, on all its leaves). Culling the node from that direction is then safe for its whole subtree. `occlusionUp` applies `InterlockedOr` word by word and is relaunched until no mask changes, like the climbs of §6.3.

*References:* code: `shaders/athenea/lod/occlusion_mask.slang` (`occlusionLeaf`, `across`, `cellOpen`, `occlusionUp`).
:::

**Conservative codebook.** The masks are replaced by an index into $N$ entries of 256 bits each. Entry 0 is all open.
A node's **worth** is the number of closed cells among the *relevant* ones, the directions with $d\cdot\hat{up} >
-0.25$ (an eye down to ~15° below the horizon). The nodes in the lowest $\log_2$-worth bins, together under 2 % of the
total worth, take entry 0 and stay out of the fit. The remaining entries are seeded from fitted nodes spread through
the file. Round 1 assigns each node to the entry nearest in Hamming distance over the relevant cells. Rounds 2–8
assign a node only to an entry that **contains** its open set, choosing the one with the fewest relevant open cells.
After each round an entry becomes the OR of its members. Every entry is therefore a superset of each member's open
set, and the codebook can never close a direction a node has open. The codebook's cost is the cells it reopens
($M_{\text{entry}}\wedge\lnot M_n$) as a share of all closed cells.

::: {.nota}
**Binary vector quantisation** — Storing 256 bits per node (32 B) for ~150 M nodes would be expensive; they are replaced by an index into a **codebook** of $N$ masks. It is *vector quantisation* (Lloyd/k-means, [23]) over binary vectors, where the natural distance is the **Hamming** distance (number of differing bits) and a cluster's centroid is not the mean but a binary mask.

**Conservative by construction** — Round 1: each node goes to the entry nearest in Hamming distance over the relevant cells (classic binary k-means). Rounds 2–8: a node may go only to an entry that **contains** its open set ($M_n\subseteq M_{\text{entry}}$), and among those it picks the most closed in relevant cells. After each round, each entry becomes the OR of its members (`occlusionUpdate`): by construction, an entry is a superset of each member, and the substitution never closes an open direction. Entry 0 is all open and always valid (a safety net).

**Worth and relevance** — A node's *worth* is the number of closed cells among the directions from which an eye can look: $d\cdot\hat{up}>-0.25$, that is, down to $\arcsin0.25\approx14.5^\circ$ below the horizon. A hood closed towards the engine (below) has many closed cells that no camera uses. The nodes are binned by $\log_2$ of their worth; the lowest bins, which together hold under 2 % of the total worth, go to entry 0 and use up no entries. The initial seeds are the masks of fitted nodes spread through the file.

**Codebook cost** — Reopened cells, $M_{\text{entry}}\wedge\lnot M_n$, as a share of the closed cells (`occlusionLost`): what culling stops discarding because of the compression.

*References:* [23] in §10; Hamming, R. W. (1950) *Error Detecting and Error Correcting Codes*; code: `shaders/athenea/lod/occlusion_mask.slang` (`relevantWord`, `closedCells`, `worthBin`, `occlusionWorth`, `occlusionSeed`, `occlusionAssign`, `occlusionUpdate`, `occlusionLost`), `modules/usd/src/Occlusion.cpp` (`occlusionBook`).
:::

**Shared by need.** One asset's trees share one codebook (`--occlusion-group`). Each unit is fitted alone at the full
size and the entries it uses are counted. The codebook is then shared out as $\text{share}_u = 16 +
\lfloor\text{spare}\cdot \text{used}_u/\sum\text{used}\rfloor$, and each unit is refitted within its share. Sharing by
fit quality and by part (`--occlusion-by-part`) were both measured and are worse:

::: {.nota}
**Sharing by need** — A single codebook for all of an asset's trees (`--occlusion-group`). First each unit is fitted alone with the whole codebook, and the number of entries it actually uses, $\text{used}_u$, is counted. Then each unit receives a minimum of 16 entries plus a proportional share of the spare ones: $\text{share}_u=16+\lfloor\text{spare}\cdot\text{used}_u/\sum\text{used}\rfloor$. Each unit is refitted within its share and its indices are offset past the earlier shares. The minimum guarantees a place even to a unit that culls nothing.

**Alternatives measured** — Sharing by fit quality favoured parts of high nominal worth and little real culling; sharing by part (`--occlusion-by-part`, a minimum of 2 entries per unit) fragments the codebook too much (66 %/74 % of the ideal culling against 84 %).

*References:* code: `modules/usd/src/Occlusion.cpp` (`occlusionMasksShared`: `least = byPart ? 2u : 16u`, `share`, `spare`), `apps/athenea/src/CmdDecimate.cpp` (`--occlusion-mask`, `--occlusion-group`, `--occlusion-by-part`).
:::

| configuration | trees | codebook | reopened | culling vs ideal |
|---|---|---|---|---|
| one codebook per tree | 6 pieces | 256 | 3.2–10 % | — |
| one fit over all trees | 6 pieces | 4096 / 1024 | 3.2 % / 6.5 % | — |
| + worth, relevant cells only | 6 pieces | 4096 / 1024 | 21.9 % / 24.7 % | — |
| worth = closed cells, not leaves | 6 pieces | 4096 | 14.7 % | — |
| shared by need | 6 pieces | 4096 | 7.2 % (grille 2.8 %, 1 249 entries) | 86 % near / 82 % far |
| 8 groups by need | car 640 | 4096 | 16.2 % | — |
| 8 groups by need | car 640 | 8192 | 11.9 % | 84 % |
| by part (192) | car 640 | 4096 / 8192 | — | 66 % / 74 % |
| 8 groups by need | car 1280 | 8187 | 12.3 % | — |
| 24 trees by need | car 5120 | 8182 | 0–23.6 % per tree | — |

::: {.nota}
**Reading the table** — "Reopened": the share of closed cells that the codebook opens again (lower is better). "Culling vs ideal": nodes dropped with the codebook relative to those the exact masks would drop.
- *One codebook per tree* (256 entries): 3.2–10 % reopened.
- *One fit over all* (4096/1024): 3.2 %/6.5 %.
- *+ worth, relevant cells only*: the fit no longer tries to keep cells below the horizon; the total over all cells rises (21.9 %/24.7 %), but not by losing useful culling (see the following note).
- *Worth = closed cells, not leaves*: weighting by leaves favoured a fender's large merges over the grille's many small bars; a node of the cut is one Gaussian drawn, however many leaves it covers.
- *Shared by need*: 7.2 % (grille 2.8 % with 1 249 entries); 86 % of the ideal culling near and 82 % far.
- *Car at 640 px, 8 groups*: 16.2 % with 4096, 11.9 % with 8192 (84 % of the ideal). By part (192 units): 66 %/74 %.
- *Car at 1280 px*: 8187 effective entries, 12.3 %. *Car at 5120 px*: 24 trees, 8182 entries, 0–23.6 % per tree. (The figures 8187/8182 rather than 8192 reflect the rounding down $\lfloor\cdot\rfloor$ of the shares.)

*References:* code: `modules/usd/src/Occlusion.cpp`.
:::

(From the third row onward, the fit counts only relevant cells, so the total reopened over *all* cells rises. Those
percentages are not comparable with the first two rows.) On the close-up level, the sum of closed cells over ~150 M
nodes exceeds a 32-bit counter, so the totals are summed in 64 bits. The per-tree results are: Windshield 0.0 %
(glass, 1 entry used), Part_2 5.3 %, Hood 7.0 %, the pilot door 7.7 %, the fine groups 8.9–15.9 % and the small-panel
group pan2 23.6 %.

::: {.nota}
**Non-comparable percentages** — From the third row on, the fit counts only relevant cells, so it can leave cells below the horizon open at no cost; the percentage over all cells rises even though the useful culling does not get worse.

**32-bit overflow** — ~150 M nodes × up to 256 cells $\approx3.8\times10^{10}$, against $2^{32}\approx4.29\times10^9$. The first total came out as "132.4 %" because the counter wrapped around; the per-tree values and the codebook were correct. The totals are now summed in 64 bits (`uint64_t closed` in `OcclusionBook`).

**Per tree** — The windscreen (glass, all open) uses one entry and reopens 0 %; the large parts reopen 5–8 %, and the groups of fine parts and of small panels, with very varied occlusion shapes, up to 23.6 %.

*References:* code: `modules/usd/include/athenea/usd/Export.h` (`OcclusionBook`: `closed` in 64 bits, `seenClosed`).
:::

**The player's rule** (and `occludedFrom` in the cut). This rule applies only when the eye is outside the asset's
bounding box. Let $d=\mathrm{normalize}(e-\mu_n)$ and $q=16\,\mathrm{oct}(d)-0.5$. The four cells
$\lfloor q\rfloor+\{0,1\}^2$, clamped to $[0,15]^2$, are the bilinear $2\times2$ footprint toward the eye. The node is
culled when all four are closed in its entry. With the car as occluder (never in a partial scene), the rule removes
12 % of the near cut and 8 % of the far cut for 0.06 dB. On the whole car at 393 px, with back culling, occlusion
takes the near cut from 522 931 to 360 333 nodes.

::: {.nota}
**The lookup rule** — With $d$ the direction from the node to the eye, $q=16\,\mathrm{oct}(d)-0.5$ places the direction in cell coordinates with the centres at integers; the four cells $\lfloor q\rfloor+\{0,1\}^2$ are those that would enter a bilinear interpolation (the $2\times2$ *footprint*), clamped to $[0,15]^2$. The node is dropped only if all four are closed: another conservative margin on top of the dilation. It is the same lookup the shading of the TX bits does (`splatCellsOpen` in `shaders/athenea/splat/splat_relight.slang`).

**Only with the eye outside the box** — The bits say whether a ray from the Gaussian reaches infinity. An eye inside the bounding box could be between the Gaussian and its occluder, and would see something the bits declare occluded. And only with the whole asset as occluder: in a partial scene (a few parts) the occluders that closed the directions are absent.

**Figures** — 12 % fewer nodes in the near cut and 8 % in the far one for 0.06 dB. On the whole car at 393 px with back-face culling, the near cut drops from 522 931 to 360 333 nodes (−31 %).

*References:* code: `shaders/athenea/lod/lod_tree_cut.slang` (`occludedFrom`, `treeErrorKeep`, `treeCulledMark`), `shaders/athenea/common/octahedral.slang` (`octEncode`), `shaders/athenea/splat/splat_relight.slang` (`splatCellsOpen`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-occlusion`).
:::

## 6.8 Pages by error

A player cannot hold a whole tree. It needs units that it fetches whole and tests without reading their nodes.

::: {.nota}
**Pages** — Fixed-size streaming units (here $P$ nodes). The player fetches them whole and decides whether it needs them by looking only at a header (box and errors), without reading their nodes: so it can manage residency at a cost proportional to the number of pages, not of nodes. It is the role of Nanite's *pages* ([35]) or of 3D Tiles' *tiles*.

*References:* [35] in §10; code: `shaders/athenea/lod/lod_pages.slang`.
:::

**Alternative considered: pages by level.** Pages taken as breadth-first tops of sibling subtrees still need 114 of
the hood's 263 pages for a far cut. The bounds keep small nodes wherever a highlight or a colour turns, so the cut
spans nodes from single leaves to $2^{11}$ leaves.

::: {.nota}
**Pages by level (discarded)** — Taking as a page the top of a subtree, walked breadth-first, and grouping siblings: the natural paging of a hierarchy. It fails because a cut of the merge tree is not at one level: the colour and normal bounds (§6.3) keep small nodes wherever a highlight turns or a colour changes, and the cut mixes single leaves with nodes of $2^{11}=2048$ leaves. A subtree page holds nodes of many sizes, and a far cut of the hood touches 114 of its 263 pages.

*References:* code: `shaders/athenea/lod/lod_pages.slang` (`treePagesTouched`, `treePagesCount`: the paging by level of `--cut-pages`).
:::

**Pages by error.** A node is needed exactly when it is a root or its parent's projected error exceeds $\tau$. So the
layout follows the parent's error (`lod_pages.slang`). The **band** of a node is

$$b(n) = \mathrm{clamp}\big(\lfloor\log_2\varepsilon(\mathrm{par}(n))\rfloor+40,\ 0,\ 63\big),\qquad b(\text{root}) = 63 .$$

::: {.nota}
**When a node is needed** — A top-down traversal reads a node to decide whether to draw it or descend to its children. It reads a node exactly when its parent is refined, that is, when $P(\mathrm{par}(n))>\tau$ (or it is a root). That is why the paging follows **the parent's error**, not the node's own.

**The band** — $b(n)=\lfloor\log_2\varepsilon(\mathrm{par}(n))\rfloor+40$, clamped to $[0,63]$: the octave of the parent's error in metres. The offset of 40 puts band 0 at $2^{-40}\approx9\times10^{-13}$ m and band 63 at $2^{23}$ m: every useful error falls inside. A root goes to band 63 (always needed). For a given $\tau$ and distance, the bands needed are those with $\varepsilon f/d>\tau$: a prefix of the bands ordered from coarsest to finest.

*References:* [35] in §10; code: `shaders/athenea/lod/lod_pages.slang` (`bandOf`, comment "PAGES BY ERROR (Nanite's order)").
:::

The sort key, in the pages' own order, is (coarsest band first, then first leaf, then larger node first):

$$\text{hi} = (63-b)\ll22 \;|\; \text{bucket}\ll19 \;|\; \text{start}\gg6,\qquad \text{lo} = (\text{start}\,\&\,63)\ll26 \;|\; (2^{26}-1-\text{leaves}).$$

::: {.nota}
**The 64-bit key** — Two 32-bit words that a GPU radix sort orders as one:
- `hi`: bits 22–27 $=63-b$ (6 bits, coarsest band first); bits 19–21 = orientation bucket (3 bits, 0 unless `--pages-by-facing`); bits 0–18 = `start >> 6` (the top 19 bits of the first leaf).
- `lo`: bits 26–31 = `start & 63` (the low 6 bits of the first leaf); bits 0–25 = $2^{26}-1-\text{leaves}$ (more leaves → smaller key → earlier).

So, within a band, nodes are in the spatial order of their leaves (depth-first order = locality), and for the same first leaf, the larger one first. `start` allows up to $2^{25}$ leaves (`min(runStart, 0x1FFFFFF)`).

*References:* code: `shaders/athenea/lod/lod_pages.slang` (`treeBandKeys`), `modules/gpu/include/athenea/gpu/algo/RadixSort.h`.
:::

A GPU radix sort orders the nodes, and page $p$ holds sorted positions $[pP,(p+1)P)$. The pages cross band
boundaries, so a coarse band of a few nodes does not become a page of its own. A parent's band is never finer than its
child's, its first leaf is never later, and when both tie the parent is the larger node. A parent is therefore never
placed after its child. (A layout that runs each band's leaves backwards puts 92 parents after their children on the
hood.) `--tree-pages P` writes, for each node, `lodPage`, and for each page:

- `lodPageInfo` = (coarsest band, nodes, first node in page order), and
- `lodPageBounds` = the box holding the **parents'** spheres (a node's need is measured from its parent) and the
  largest and smallest parent error.

::: {.nota}
**Parent before child** — So that a traversal can start with the coarse pages without waiting for the fine ones, a parent must never come after its child in the order. This holds because, comparing keys: (i) the parent's band is never finer (the grandparent's error $\ge$ the parent's, by monotonicity); (ii) if they tie, the parent's first leaf is never later (the parent's run contains the child's); (iii) if those tie too, the parent has more leaves and wins the tie-break. Walking the leaves backwards within a band broke (ii): 92 parents behind their children on the hood.

**Pages that cross bands** — Page $p$ is positions $[pP,(p+1)P)$ of the total order, not aligned to bands: a coarse band with few nodes does not take up an almost empty page.

**Page table** — `lodPageInfo` (coarsest band, number of nodes, first node) and `lodPageBounds`: a box holding the spheres of its nodes' **parents** (need is measured from the parent), and the largest and smallest parent error. The boxes are accumulated with `InterlockedMin`/`InterlockedMax` on floats converted to an orderable integer (`ordered`: inverts the negatives and sets the sign bit of the positives).

*References:* code: `shaders/athenea/lod/lod_pages.slang` (`treeBandKeys`, `treeBandTable`, `treeBandTableOut`, `ordered`, `unordered`), `modules/usd/src/Decimate.cpp` (writing `lodPage`, `lodPageInfo`, `lodPageBounds`), `apps/athenea/src/CmdDecimate.cpp` (`--tree-pages`).
:::

The player projects the largest error from the box and loads the page when that projection exceeds $\tau$. The
bucket bits (`--pages-by-facing`: dominant normal axis and sign) make residency worse, 4.45× against 3.98× at
$\tau=2.5$, and are off.

::: {.nota}
**The player's test** — Projects the page's **largest** parent error from the nearest point of its box; if it exceeds $\tau$, some node of the page might be needed and the page is loaded. It is a conservative bound: it never drops a needed page, at the price of loading some that are not.

**Orientation buckets** — Grouping within each band by the dominant normal axis and its sign (6 buckets) was meant to let back-face culling drop whole pages. It makes residency worse (4.45× against 3.98× the cut, at $\tau=2.5$): splitting by orientation breaks spatial locality, and each view still needs a part of nearly every bucket. Off.

*References:* code: `shaders/athenea/lod/lod_pages.slang` (`treeBandKeys`, `facingBuckets`), `apps/athenea/src/CmdDecimate.cpp` (`--pages-by-facing`).
:::

**Residency.** A walk needs the cut and all its ancestors, about 2× the cut. Pages hold about 4× the cut whatever
their size or order, because needed and culled nodes share every page while the culling depends on the view.

| configuration | scene | $\tau$ | cut | nodes needed | pages holding (× cut) |
|---|---|---|---|---|---|
| P 3000 | six pieces, near 1920 | 1 | 244 114 | — | 2.0–2.6× |
| P 3000 | six pieces, far 393 | — | — | — | 2.2–4.9× (hood: 2 pages) |
| P 3000 | car 640, near 393 | 2 | 293 629 | 647 707 (2.20×) | 1 170 000 (3.98×) |
| P 3000 | car 640, near 393 | 2.5 | 263 918 | 580 997 (2.20×) | 1 050 000 (3.97×) |
| P 3000 | car 640, near 393 | 3 | 198 702 | 445 195 (2.24×) | 957 000 (4.81×) |
| P 3000 | car 640, near 393 | 6.75 | 92 987 | 214 827 (2.31×) | 423 000 (4.54×) |
| P 1000 / 500 | car 640, near 393 | 2 | 293 629 | — | 3.91× / 3.87× |
| P 1000 / 500 | car 640, near 393 | 2.5 | 263 918 | — | 3.77× / 3.60× |

::: {.nota}
**Why ~2× the cut** — A traversal needs the cut and all its ancestors. In a binary tree, the ancestors of a set of nodes are at most as many as the set (each internal node of the "upper tree" has two children), so the total comes to about $2\times$: 2.20–2.31× in the table.

**Why ~4× in pages** — Pages group by error band and by position, but culling (back faces, occlusion) depends on the view. Each page mixes needed nodes with nodes this view drops, and it has to be fetched whole. The factor is about 4× whatever the size ($P$ 3000, 1000, 500) or the order.

**Reading the table** — 640 px car, close view at 393 px: at $\tau=2$ the cut has 293 629 nodes, the traversal needs 647 707 (2.20×) and the pages holding them add up to 1.17 M (3.98×). At $\tau=6.75$ the cut drops to 92 987 and the pages to 423 k (4.54×). Smaller pages help little (3.91×/3.87× with $P=1000/500$ at $\tau=2$). On the six pieces, 2.0–2.6× near and 2.2–4.9× far (the whole hood in 2 pages).

*References:* code: `shaders/athenea/lod/lod_pages.slang` (`treeBandPages`, `treeErrorPagesCount`, `reachable`).
:::

`treeCulledMark` marks the nodes the culls leave out. A culled node's subtree is culled too, so a walk stops there.
That brings the nodes needed to 2.2× (a walk into culled subtrees needs ~4×), but the pages still hold 3.6–4.5×. A
streaming player can therefore keep the traversal at 12 B per node for what is needed and the payload in deflate runs
of 64 for what is drawn; in the WebGPU player of §7, near at $\tau=3$, that comes to ~60 MB and 226 k drawn at 60 fps
(emulated).

::: {.nota}
**`treeCulledMark`** — Marks the nodes that the culls (back face on a closed shell, occlusion) leave out from the view. Because the cone and the mask bound the whole subtree, a traversal stops at the marked node; `reachable` in `lod_pages.slang` checks that no ancestor is marked when counting the needed nodes. That keeps the needed nodes at 2.2× (without the mark, a traversal that enters culled subtrees needs ~4×), but the pages stay at 3.6–4.5×: paging is view-independent.

**Separating traversal and payload** — The traversal needs per node only what decides: centre, radius, error, links (12 B per node in the text's figure, with quantisation). The payload (shape, colour, transfer) is needed only for what is drawn and can go in deflate-compressed (RFC 1951) runs of 64 nodes. So residency of the heavy part follows the cut, not the pages. In the WebGPU player of §7, near at $\tau=3$: ~60 MB and 226 k Gaussians drawn at 60 fps (on an emulated phone).

*References:* [46] in §10; Deutsch, P. (1996) RFC 1951 *DEFLATE Compressed Data Format*; code: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCulledMark`), `shaders/athenea/lod/lod_pages.slang` (`reachable`). The player is sparkwebGPU (Appendix A.1; demos at https://sparkwebgpu.lucab.co.uk) and is not part of the engine's repository.
:::

## 6.9 The close-up level

Close up, the logo and grille stay soft at the 1280 px bake. A close camera comes to about 0.5 mm per output
pixel, while the 1280 px leaves are 3–4 mm. A logo-and-grille bake from a close camera at 5120 px shows the checkered
flag, the bowtie and the grille's mesh crisp (108 527 → 404 594 Gaussians). The close-up level (car5120) re-bakes the
car in groups (Figure 10):

- **Detailed parts.** The 66 parts that the 1280 px bake gives at most 1.5 Gaussians per triangle (badges, grilles,
  lamps, calipers, mirrors, wheels, wipers, trims), at 5120 px, in 7 groups (`fine0–6`).
- **Panels.** Each part at 5120 px, or as fine as keeps it under a leaf cap of 4.5 M (a 5.01 M-leaf tree is at the
  limit of device memory when written), with the small ones grouped up to 4 M. `--max-cells 1048576` is required: at
  4736 px, Infinite_Black loses 359 628 cells past the default 262 144 per triangle, which leaves holes.

::: {.nota}
**Why a close-up level** — At 1280 px width of the conversion camera, the leaves are 3–4 mm on the hood; a close camera on the phone reaches ~0.5 mm per pixel. A factor of 6–8 in size needs ~4× more linear resolution: 5120 = 4 × 1280, and ~16× more Gaussians where the geometry allows. The logo-and-grille test went from 108 527 to 404 594 Gaussians (3.7×) and showed the checkered flag, the bowtie (Chevrolet's emblem) and the mesh crisp.

**Detailed parts** — The 66 parts with at most 1.5 Gaussians per triangle at 1280 px are limited by tessellation, not by resolution: at 1280 px a couple of Gaussians per triangle already fit, which is why they gain most from 5120 px (each triangle is subdivided into more cells).

**Panels and the 4.5 M cap** — A tree of 5.01 M leaves is at the limit of device memory when written (§6.9, below). Each panel is converted at 5120 px or at the highest resolution that keeps it under the cap (hence 3392, 4480, 4736 px in the table).

**`--max-cells`** — `mesh2splat` walks at most `maxCells` cells per triangle (by default $2^{18}=262\,144$, `CmdMesh2Splat.cpp`); the rest is not sampled and a warning is printed. Infinite_Black, with very large triangles, lost 359 628 cells (holes) at 4736 px; with $2^{20}=1\,048\,576$ it was redone in full.

*References:* [12], [13] in §10; code: `apps/athenea/src/CmdMesh2Splat.cpp` (`maxCells = 1u << 18`, the warning "cells lay past --max-cells").
:::

Each group is converted with every other part hidden, but the bake traces the whole stage, so a part keeps the others'
shadows. Trees are per part, so the close-up level has the same format as car1280: the coarse roots stay, and the
deeper levels are paged in only close up.

::: {.nota}
**Isolated conversion, global bake** — `--hide` affects only the conversion: the bake traces the whole stage, so each part keeps the shadows and interreflections of the others even though it is converted alone. That is what allows splitting into groups without shadow artefacts.

**Same format as car1280** — Because the trees are per part (the key guard), the close-up level is one more set of trees with the same schema: its coarse roots stay resident and the fine levels are paged in only when the camera comes close (by projected error, §6.8).

*References:* code: `apps/athenea/src/CmdMesh2Splat.cpp` (`--hide`).
:::

| groups | leaves | resolution |
|---|---|---|
| fine0–6 (66 detailed parts) | 18 610 627 | 5120 |
| door (pilot) | 4 084 170 | 5120 |
| Bottom | 4 897 428 | 3392 |
| Part_3 | 4 370 459 | 4480 |
| Infinite_Black | 4 481 738 | 4736 |
| Panel_under_Door_001, Windshield, Front_Fender, Hood, Front_Bumper, Part_2, Part_23 | 15 994 566 | 5120 |
| pan0–4 (small panels) | 17 088 152 | 5120 |
| Front_Bumper_001 | 5 008 986 | 4480 |
| **24 trees** | **74 536 126 leaves, 74 328 031 merges, 208 095 roots** | |

::: {.nota}
**Reading the table** — 24 trees: 7 of fine parts (18.6 M), the pilot door, Bottom, Part_3, Infinite_Black, 7 large parts at 5120 px (16.0 M), 5 groups of small panels (17.1 M) and Front_Bumper_001. The leaves add up to 74 536 126.

**Check** — In a binary forest, $\text{roots}=\text{leaves}-\text{merges}=74\,536\,126-74\,328\,031=208\,095$, as the table says. The nodes are $74\,536\,126+74\,328\,031=148\,864\,157$ (next paragraph).
:::

The 24 trees hold 148 864 157 nodes in 49 634 pages of 3000, and 87.4 GB on disk. Rebuilt from their leaves with the
second moments about each group's own centre (§6.2.1), and with a shared occlusion codebook of 8181 entries (11.9 % of
cells reopened), the car's cut at 0.25 px draws 11.7 M Gaussians instead of 16.9 M and reads 24.00 dB on the frame and
18.31 dB on the car against the path trace (23.62 and 17.91 before). Some trees keep many lone roots
(fine5: 29 213; Infinite_Black: 92 773), because merges never cross parts or guards, by design. The player's
packer (§7.5.4) adds a multi-level synthetic root above them, since a file holds about 3 000 roots.

::: {.nota}
**Pages and disk** — $148\,864\,157/3000\approx49\,621$ pages if everything were a single order; with 24 trees and the last page of each incomplete, a few more come out, 49 634. 87.4 GB / 148.9 M nodes ≈ 590 B per node: the TX transfer, the shadow bits and the other attributes dominate over the geometry.

**Rebuilt after the fix of §6.2.1** — With the centred second moments and the shared codebook of 8181 entries (11.9 % reopened), the cut at 0.25 px draws 11.7 M instead of 16.9 M (−31 %) and improves by +0.38 dB on the frame and +0.40 dB on the car against the path trace.

**Lone roots** — fine5 keeps 29 213 roots and Infinite_Black 92 773: the guards (key, glass, mirror) stay active even in the relaxed top, by design. Since a file holds about 3 000 roots, the player's packer (§7.5.4) adds a multi-level synthetic root on top, so that the traversal starts at a single node.

*References:* `docs/decisions.md` ("A merge's second moment about its own centre (13 Oct)": "the 5120 px trees are to be rebuilt from their bakes").
:::

**Memory for trees of 4–5 M leaves.** A front wheel with 4.45 M leaves and 8.88 M nodes (the tyre alone 2.56 M) builds
its tree within device memory, but writing it needs two changes.

1. `mergeAttribute` needs a weights buffer only for a rig's joints. Allocated as large as each attribute, it costs
   1.7 GB alongside each of a transfer's 48-float arrays over 8.9 M nodes. It is one word for every other attribute.
2. The write must not hold each array three times in unified memory (what was read, a copy, and the merged arrays
   started as another copy). The merged arrays start from the metadata alone, and the tree's write, the last step,
   releases each array once it is on the device.

With these changes, a 4.08 M-leaf door peaks at 10.5 GB resident and 4.90 M leaves are written. Front_Bumper_001
(5.01 M) is written with 2.23 GB of free memory at the lowest point; the 4.5 M cap for panels keeps that margin.

::: {.nota}
**1. The weights buffer** — `Decimator::mergeAttribute` (`modules/lod/src/Lod.cpp`) merges one attribute over runs of leaves. Only the joints mode (mode 3, a rig's joints) writes weights alongside the values. Previously a weights buffer the size of each attribute was allocated: $48\ \text{floats}\times8.9\,\text{M nodes}\times4\,\text{B}\approx1.7$ GB for each transfer array. Now it is a single word except for joints.

**2. Three copies in unified memory** — On Apple Silicon the GPU and the CPU share physical memory: what the host holds competes with the device. The write held what was read, a copy, and the merged arrays started as another copy. Now the merged arrays start from the metadata, and the write releases each array as soon as it is on the device.

**Results** — A 4.08 M-leaf door peaks at 10.5 GB resident. Front_Bumper_001 (5.01 M) is written with 2.23 GB free at the lowest point; the cap of 4.5 M leaves per panel keeps that margin.

*References:* code: `modules/lod/src/Lod.cpp` (`Decimator::mergeAttribute`, comment "Only a rig's joints (mode 3) write weights"), `modules/lod/include/athenea/lod/Lod.h` (`mergeAttribute`), `modules/usd/src/Decimate.cpp`.
:::

**Occlusion written in place.** At 87.4 GB for the 24 trees, a copy of each tree does not fit beside them on disk.
`occlusionMasksShared` writes each tree to `<tree>.occlusion.usdc` beside it and renames it over the original. The
shared codebook (8 182 entries) takes 645 s, with a peak resident set of 10.2 GB; per tree it reopens 0–23.6 % of the
closed cells (§6.7).

::: {.nota}
**Writing in place** — `occlusionMasksShared` writes each tree with its masks to `<tree>.occlusion.usdc` and renames it over the original. A rename within one file system is atomic (POSIX `rename`): at no point do two complete copies of the 87.4 GB coexist, and a failure leaves the original intact.

**Cost** — 645 s and a 10.2 GB peak for a shared codebook of 8 182 entries over the 24 trees; each tree reopens 0 to 23.6 % of its closed cells (§6.7).

*References:* code: `modules/usd/src/Occlusion.cpp` (`occlusionMasksShared`), `modules/usd/include/athenea/usd/Export.h`, `apps/athenea/src/CmdDecimate.cpp`.
:::

## 6.10 Measured results

All renders are the engine's rasteriser. "Leaves at 4×" means every leaf drawn at 4× resolution and box-reduced, the
reference for far cuts (a 1920 px bake drawn at 393 px is see-through). "PT" is the mesh path trace at 1920×1080 from
`/World/Camera`, compared on the frame and on the car's 900×420 crop.

::: {.nota}
**"Leaves at 4×"** — All leaves drawn at 4× the output resolution and reduced with a box filter ($4\times4$ pixels → 1): a supersampled render, the reference for far cuts. It is needed because a bake sized for 1920 px, drawn at 393 px, leaves each Gaussian far below a pixel and the image "turns see-through" (the aliasing and scale problem of 3DGS addressed by Mip-Splatting, [4]). Comparing a cut against the leaves at native resolution would measure that defect, not the LOD's.

**"PT"** — The path trace of the original mesh (§2), 1920×1080 from `/World/Camera`: the reference truth of the whole pipeline. It is measured on the full frame and on a 900×420 crop that frames the car.

*References:* [4] in §10; code: `apps/athenea/src/CmdCompare.cpp` (`athenea compare`); `docs/operations.md` §2.11 "`athenea compare` — what an image holds, measured on the GPU".
:::

**The 640 px car at 393 px, against the leaves at 4×.**

| configuration | cut | near nodes | near dB | far nodes | far dB |
|---|---|---|---|---|---|
| coverage trees | error 1 px, bounds | 666 585 | 33.29 | 343 541 | 39.15 |
| mass trees | error 1 px, bounds | 666 585 | 33.27 | — | — |
| coverage trees | $\tau=0.5$ | 930 119 | 33.17 | — | — |
| + back culling | closed shells by the 2 % rule | 522 931 | 33.63 | — | — |
| + occlusion | 8192 shared by group | 360 333 | 33.58 | 195 049 | 39.53 |
| both culls | $O<2\sqrt{E}$ rule (102 parts) | 378 028 | 33.53 | — | — |

::: {.nota}
**Reading the table** (640 px car, output 393 px, against the leaves at 4×):
- *Coverage trees, error 1 px with bounds*: 666 585 nodes near at 33.29 dB; 343 541 far at 39.15 dB.
- *Mass trees*: the same nodes, 33.27 dB: opacity hardly matters at this scale.
- *$\tau=0.5$*: 930 119 nodes at 33.17 dB: lowering the threshold without bounds adds nodes without gaining quality.
- *+ back-face culling* (2 % rule): 522 931 nodes and +0.34 dB (the net of cracks disappears).
- *+ occlusion* (8192 entries shared by group): 360 333 near (−31 %) at 33.58 dB and 195 049 far at 39.53 dB.
- *Both culls with the $O<2\sqrt E$ rule*: 378 028 nodes at 33.53 dB; slightly more than with the 2 % rule because fewer parts (102 against 111) are closed shells, and the light between the spokes is fixed.

*References:* code: `shaders/athenea/lod/lod_tree_cut.slang`.
:::

The far cut (195 k) fits a budget of ~250 k nodes, the one the mobile player of §7 draws at 30 fps. The near cut is
~1.4× that.

::: {.nota}
**The mobile player's budget** — ~250 k nodes is what the player of §7 draws at 30 fps on the reference phone. The far cut (195 k) fits; the near one (360 k) exceeds it by ~1.4×. The player sets its draw target by frame time, so close up it draws a somewhat coarser cut.

*References:* [11], [46] in §10.
:::

**Against the path trace (1920 px).**

| bake | what | nodes | car crop | frame |
|---|---|---|---|---|
| 640 px | all leaves | 5.38 M | 16.37 | 21.92 |
| 640 px | error 1 px, both culls | 675 760 | 16.28 (31.91 vs leaves) | 21.80 |
| 640 px | error 0.25 px, back culling | 2 313 541 | 16.29 (34.01 vs leaves) | 21.81 |
| 640 px | leaves, unwidened sharp lobes | 5.38 M | 16.58 | 22.15 |
| 640 px | 0.25 px, back culling, shading fixes | — | 16.55 | 22.18 |
| 640 px | leaves (0.01 px), no cull / cull | 4.35 M / 3.70 M | 16.73 / 16.75 | 22.40 / 22.43 |
| 640 px | group 0 with spill $w_s=2$ | 561 242 (g0) | 16.79 | 22.38 |
| **1280 px** | 0.25 px, back culling | 4 998 976 | **17.88** | **23.90** |
| 1280 px | 0.25 px, back culling, tube/fold split (§4.9.4) | — | 17.88 | 23.91 |

::: {.nota}
**Reading the table** (against the path trace at 1920 px; PSNR of the car crop and of the frame):
- *All leaves at 640 px*: 5.38 M Gaussians, 16.37 / 21.92 dB: the ceiling the LOD can reach with that bake.
- *Error 1 px, both culls*: 675 760 nodes (1/8 of the leaves), 16.28 dB, only 0.09 dB less; against the leaves themselves, 31.91 dB.
- *Error 0.25 px, back-face culling*: 2.31 M nodes, 16.29 dB (34.01 against leaves).
- *Unwidened sharp lobes* and *shading fixes* (§4): improvements to the bake/shading, not to the LOD (16.58, 16.55).
- *Leaves (cut at 0.01 px), without/with cull*: 4.35 M / 3.70 M; 16.73 / 16.75 dB. The cut at 0.01 px is practically the set of leaves; that the count falls below 5.38 M indicates that some merges (for example, of nearly coincident Gaussians) have a projected error under 0.01 px and replace their leaves.
- *Group 0 with spill* (§6.5): 16.79 dB.
- *1280 px, 0.25 px with back-face culling*: 5.0 M nodes, **17.88 / 23.90** dB: the jump comes from the bake.
- *Tube/fold split* (§4.9.4): +0.01 dB on the frame.

**Low PSNR in absolute terms** — 16–24 dB against the path trace are not LOD errors: they compare two different renderers (baked Gaussians against a path trace of the mesh) and include differences in highlights, lamps and sharpness of reflections. What measures the LOD is the difference between rows with the same bake.
:::

The level of detail costs about 0.1 dB against the path trace (16.37 → 16.28 at 1/8 of the leaves). The bake accounts
for the rest: its resolution, its lamps and the sharpness of its reflections. The 1280 px re-bake (8.69 M leaves,
8.67 M merges, 20 723 roots, 5 789 pages, occlusion reopening 12.3 %) adds 1.3 dB on the car. Against the 640 px
0.25 px cut with the shading fixes, the 1280 px cut takes the grille's holes from 2.24 to 1.04 of the path trace, the
hood highlights from 3.2 to 2.0 px off, and the joints from 0.90 to 1.02 of the path trace.

::: {.nota}
**The cost of the LOD** — 16.37 → 16.28 dB with 675 760 nodes out of 5.38 M (12.6 %, ~1/8): about 0.1 dB. The rest of the error against the path trace belongs to the bake: resolution, lamps (the headlights' lights were not in the bake) and sharpness of reflections.

**The 1280 px bake** — 8.69 M leaves, 8.67 M merges, 20 723 roots ($8.69\,\text{M}-8.67\,\text{M}\approx0.02$ M, consistent), 5 789 pages of 3000 (≈17.36 M nodes / 3000) and 12.3 % reopened in occlusion. It gains +1.33 dB on the car (16.55 → 17.88). Grille holes: from 2.24 to 1.04 of the path trace (1.0 would be exact). Hood highlights: from 3.2 to 2.0 px of position offset. Joints: from 0.90 to 1.02 of the path trace (too dark before, almost exact now).
:::

## 6.11 Limits and open items

- **Projection-only holes.** A coarse cut of sub-pixel strips (the grille) fills holes that no per-node, view-free
  measure can see (§6.5). This is resolved in practice by finer leaves, not by the cut.
- **Occlusion weighting.** The codebook is shared by entries used, not by how often a view culls a node. Weighting by
  an orbit of views in the bake is expected to lift culling past 84 % of the ideal.
- **Residency at ~4×.** It is bound by pages that mix needed and view-culled nodes. A streaming player splits traversal from
  payload rather than shrinking pages.
- **Glass merges.** Every thin-sheet Gaussian is lifted to the 0.1 sheet floor, so merging fewer, larger sheets varies
  the transmission cell to cell. Spark declares sheet merges at 4× their size in the interim, and a floor per pixel
  rather than per Gaussian is pending.
- **Far, thin trims** still become fat dashes in every cut. The bounds miss this error.
- **Not ported.** The closed-shell back rule and the error cut live in a development branch. The main engine's
  `CutSelector` is unchanged.

::: {.nota}
**Projection-only holes** — See §6.5: a defect that exists only on screen cannot be measured by a view-independent error. The possible ways out are finer leaves (the one adopted) or a view-dependent measure computed in the cut itself.

**Occlusion weighting** — Codebook sharing uses entries used as its measure of need. What matters is how often a real view drops a node; weighting by an orbit of cameras during the bake would approximate that frequency (as `--drop-hidden` does for visibility, `lod_prune.slang`).

**Residency at ~4×** — Shrinking pages does not reduce it (§6.8: 3.6–3.9× with $P=500$). The way out is to separate the traversal, which is compact, from the payload, which is fetched in runs according to what is drawn.

**Glass merges** — Thin glass is drawn as a sheet with a minimum opacity of 0.1 (`kSheetAlpha` in `shaders/athenea/splat/splat_project.slang`). When sheets are merged into fewer, larger ones, that per-Gaussian floor changes the total transmission from one region to another. Spark compensates by declaring sheet merges at 4× their size; a per-pixel floor instead of a per-Gaussian one is what remains pending.

**Far, thin trims** — A sub-pixel chrome trim turns into a fat dash: the colour and normal bounds do not see the width error of a thin linear object (the same shape problem as the needle guard, off by default).

**Not ported** — In the engine's repository, the error cut with bounds (`treeErrorKeep`, `treeBoundsUp`), back-face culling of closed shells (`--cut-cull-back`, buffer `cullable`), occlusion and pages exist in the offline tool `athenea decimate` (`modules/usd/src/Decimate.cpp`, `shaders/athenea/lod/lod_tree_cut.slang`, `lod_pages.slang`, `occlusion_mask.slang`), and the $O<2\sqrt E$ rule in `apps/athenea/src/CmdMesh2Splat.cpp`. What has not changed is the engine's `CutSelector` (`modules/lod/src/Lod.cpp`, `shaders/athenea/lod/lod_cut.slang`), which is still the size cut over the octree of §6.1: the engine's real-time render does not yet walk merge trees. Real-time tree traversal lives in the sparkwebGPU player (§7), outside this repository.

*References:* code: `modules/lod/src/Lod.cpp` (`CutSelector`), `shaders/athenea/lod/lod_cut.slang`, `shaders/athenea/lod/lod_prune.slang`, `shaders/athenea/splat/splat_project.slang` (`kSheetAlpha`); `docs/decisions.md` ("Levels of detail" → "Not done yet").
:::
