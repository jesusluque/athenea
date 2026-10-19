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

Notation. A node $n$ is a leaf (a baked Gaussian) or a merge. $\mu_n$ is its centre and $\Sigma_n$ its covariance,
with scales $s_n = (s_x, s_y, s_z)$ and opacity $o_n$. $\mathrm{par}(n)$ is its parent. $\tau$ is the threshold in
pixels, $f$ is the focal length in pixels at unit distance (`pixelsPerUnit`), and $e$ is the eye.

## 6.1 The engine's original level of detail

**Octree by Morton code.** `LodBuilder` (`modules/lod`, `shaders/athenea/lod/lod_common.slang`) sorts the splats by a
30-bit Morton code. The octree cell of a splat at level $r$ is the top $3r$ bits of its code, so the cells of a level
are runs of equal prefixes. A boundary pass and a prefix sum find them, and no tree is walked. Each cell carries
moments, following Kerbl et al. 2024. With $w = o \cdot A_2(s)$, where $A_2$ is the area of the two longest axes,

$$W=\sum w,\quad S_1=\sum w\,\mu,\quad S_2=\sum w\,(\Sigma+\mu\mu^\top),\quad C=\sum w\,c .$$

Moments add, so each coarser level is summed from its children. A cell becomes one Gaussian with $\mu=S_1/W$,
$\Sigma=S_2/W-\mu\mu^\top$ (Jacobi sweeps in the shader), $c=C/W$ and $o=\min(W/A_2,0.99)$. The stored levels run from
`coarsestLevel` down to the deepest level whose cell count is at most half the splat count.

**Cut.** `CutSelector` draws a group when its cell's edge, projected at the cell's nearest distance, is at most $\tau$
and its parent cell's projection is above $\tau$. A child cell lies inside its parent, so the test is monotone along
any root-to-leaf line, and every place is drawn at exactly one level. The only readback is a count per level and a
read per instance per frame.

**Chunks, `.athc` and streaming.** The splats are cut into chunks of 65 536 consecutive Morton-ordered splats. The
merged levels stay on the device, and the chunks are streamed. The `.athc` format (`.lrtc` in the lucabRTrender tree)
has a 104 B header, level and chunk tables, and blocks aligned to 4 KiB so that a chunk can be mapped and faulted in
alone. `StreamingPool` maps the file. Two loader threads copy the wanted chunks, most wanted first. Each slot is taken
from a free slot, from the chunk wanted least recently, or from a chunk wanted less than half as much. A finest-level
group whose chunks are missing draws its own merged Gaussian, so the image never has a hole. On train_7k (742 k
splats, M5 Pro), the build took 47 ms and produced 146 k merged Gaussians. At 8 px the cut drew 83 k splats (11 %),
rendering in 3.3 ms against 12.6 ms, with a mean absolute error of $3\cdot10^{-3}$. On train_30k, the `.athc` streamed
into 4 of 16 slots wrote byte-identical EXRs and settled in 2 cuts.

**Why it does not scale to baked assets.** Four properties of this structure do not fit clouds of tens of millions
of material Gaussians under a shared budget.

1. The threshold is set per instance and is not a budget. Each instance costs two blocking submits and a readback per
   frame, and no budget is shared across instances.
2. The octree cells ignore the object. A cell merges whatever falls inside it, including a badge with the panel
   beneath it and glass with paint.
3. A streamed splat costs 132 B, against ~55 B for the compact record (§7.4).
4. Only the splats are chunked. The merged levels, down to one with up to half as many cells as splats, are always
   resident, and for millions of leaves they alone exceed the budget.

## 6.2 Per-part pair-merge trees

`athenea decimate --tree` builds a binary merge tree by mutual nearest-neighbour pair merging. It is NanoGS (Xiong et
al. 2026) on the device (`lod_pairs.slang`, `lod_pairs_final.slang`, `Lod.cpp`), with guards that suit baked material
clouds (Figure 16).

![Figure 16. A merge tree and its cut. Nodes are sums of moments; the error ε is monotone up the tree and each node’s sphere ρ covers its children’s. Nodes are paged by their parent’s error and carry an index into a conservative occlusion codebook; the per-frame traversal tests pages, back faces and occlusion.](figures/fig-lod.svg)

### 6.2.1 Nodes as moments

A node is a vector of 24 floats: $W$, $W\mu$, $W(\Sigma+(\mu-\bar\mu)(\mu-\bar\mu)^\top)$ (6), $W\cdot\mathrm{sRGB}(c)$, $W n$, $W\alpha$,
$W\cdot$metallic, $W\cdot$transmission, $W\cdot$specular, the key, the record count, the weighted thickness ratio and
the node's index in the tree. Here $w = o\cdot\max(s_xs_y, s_ys_z, s_xs_z)$ (`discArea`), and $\alpha$ is the GGX
roughness squared. A merge sums the two vectors. Because sums are associative, a node's moments are those of all its
leaves whatever the merge order.

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

The merged Gaussian (`pairFinalize`) takes the moment-matched $\mu$ and $\Sigma$, eigen-decomposed by Jacobi with the
thinnest axis third. Merging by moments preserves the second moment. A $2\times2$ block of discs as wide as their
spacing therefore merges into one disc 0.58 of the spacing wide, which leaves visible holes when seen up close.
Along each wide axis $u$, the merge is widened as follows. Let $v_u$ be the area-weighted variance of the member
centres, $c_u$ their mean width, and $d^2 = 12\sqrt{v_uv_v}/(n-1)$ the spacing of a grid of $n$ members over that
spread. Then

$$s_u' = \mathrm{clamp}\!\left(\frac{c_u}{d}\sqrt{12v_u+d^2},\; s_u,\; 2s_u\right).$$

The opacity under the default rule is the members' area-weighted mean, capped at 0.99. The mass coverage $W/A(s')$ is
kept as `coverageMass`. A merge of discs whose thinnest axis is under 0.12 of its middle axis is written as a surfel
(thinnest scale exactly 0).

### 6.2.2 The cost

For nodes $a, b$ with $\pi = W_a/(W_a+W_b)$ and merged $\Sigma$, the shape term is Runnalls' KL bound per unit mass.
The disc's area replaces the volume, which a surfel does not have:

$$c_{\text{shape}} = \max\!\Big(0,\ \tfrac12\big[\log I_2(\Sigma) - \pi\log I_2(\Sigma_a) - (1-\pi)\log I_2(\Sigma_b)\big]\Big),$$

where $I_2$ is the second invariant (sum of pairwise eigenvalue products, computed without eigenvalues). The full cost
is

$$c = \sigma_{\text{px}}\big(c_{\text{shape}} + w_f\,\phi^2 + w_n(1-\cos\theta_{ab})(1+F_0/\alpha)\big) + w_c\,\lVert\Delta c_{\text{sRGB}}\rVert^2 + w_m\,\Delta_{\text{mat}},$$

where $\phi$ is the thickness the merge adds over its thicker member. The term $\sigma_{\text{px}} =
\mathrm{smoothstep}(0.5, 2, \text{px})$ removes the shape terms of merges that are under half a pixel on the
conversion camera. Defaults: $w_c=20$, $w_n=1$, $w_m=4$, $w_f=4$.

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

### 6.2.4 Passes and the relaxed top

In each pass, every node computes the cost of merging with the 8 neighbours on either side along a Morton order of
the centres. It then repeats this along a second order shifted by half the finest cell, and keeps the cheaper partner.
Pairs that chose each other are binned into a $\log_{10}$ cost histogram, and the host picks the cost threshold that
merges as many pairs as the pass may, which is at most half the nodes. Merging stops at `treeRatio` (4096×) or when
nothing more may merge. With `relaxTop`, merging then continues with the colour edge (2.0), the angle, the
glass-among-itself guard, the aspect and the pixel limit lifted. The key, glass-with-non-glass and mirror guards stay.
Each such merge records its cost **negated**, which marks it as past the guards. This gives a far view coarse levels
to cut to: one headlight cover goes from 76 486 roots to 123.

**Layout.** The tree is written as a ParticleField. The leaves come first, in depth-first order, so that every node is
a contiguous run of leaves. The merges follow in the order they were made. Each node carries `lodParent` (a root is
its own parent), `lodStart` (its first leaf), `lodSize` (its number of leaves) and `lodCost`.

### 6.2.5 Many parts: `--passes`, `--merge-glass`, `--object-groups`

On the whole Corvette (5.38 M leaves), the tree can be built in one pass, but writing its attributes exceeds device
memory. `--object-groups K --object-group i` keeps the Gaussians whose `cryptoObject` mod $K$ is $i$. Merges never
cross parts, so each group's tree is exactly the subtree the whole asset would hold for those parts. Each pass merges
at most half of what is left, and a group of many small parts merges few pairs at once. The default cap of 60 passes
leaves a group with 116 549 roots out of 323 613 leaves. At `--passes 1000` it reaches 1 298 roots in 310 passes.
`--tree-report` shows two causes of the remaining roots: solid glass never merges (roots equal leaves), and the tyres
merge a trickle per pass until the cap.

| group (640 px car) | roots before | with `--merge-glass --passes 4000` |
|---|---|---|
| group 2 | 190 490 | 4 237 |
| group 5 | 36 508 | 817 |
| whole car (8 groups) | — | 17 133 roots, 5.38 M leaves, 3 582 pages |

## 6.3 The cut by error, with bounds

**Alternative considered: the size cut.** `--cut-pixels` keeps a node when its on-screen width $2\max(s)f/d$ is at
most $\tau$ and no ancestor's is. Merges past the guards pass only at $0.25\tau$. That cut is still available. The
error cut below, after Nanite, gives higher PSNR with fewer nodes (table at the end of this section).

**Own error.** Let $r_n=\lVert s_n\rVert$ (the square root of $\mathrm{tr}\,\Sigma_n$) and $\alpha_n =
\max(\text{roughness},0.02)^2$. A merge's own error in metres is

$$\varepsilon_0(n) = r_n\Big(|\mathrm{cost}_n| + w_c\,s_c(n) + w_n\,\frac{\theta(n)^2}{\alpha_n} + w_s\max(E_n-E_0,0)\Big),\qquad w_c=10,\ w_n=1,$$

and $\varepsilon_0=0$ for a leaf. Merges past the guards ($\mathrm{cost}_n<0$) are not measured by their cost, which
misses the moiré it causes on glass and the colour it drags across an edge. They are therefore held to

$$\varepsilon_0(n) \leftarrow \max\big(g\,\varepsilon_0(n),\ 2r_n\big),\qquad g=4\ (\texttt{--cut-guard}).$$

The spill term $w_s$ is 0 by default (§6.5).

**Bounds.** A merge's cost covers only the last pair it joined. A chain of merges, each within the colour edge, can
drift across that edge. So a colour spread $s_c$ and a normal cone $\theta$ are climbed up the tree by the triangle
inequality (`treeBoundsUp`), starting at 0 for the leaves:

$$s_c(p) = \max_{k\in\mathrm{ch}(p)}\big(s_c(k)+\lVert c_k-c_p\rVert\big),\qquad \theta(p) = \max_{k}\big(\theta(k)+\angle(n_k,n_p)\big).$$

Both are therefore bounds on every leaf under $p$.

**Monotone error and covering spheres.** The error is made monotone, and every node gets a sphere that contains its
children's spheres:

$$\varepsilon(n) = \max\big(\varepsilon_0(n),\ \max_{k}\varepsilon(k)\big),\qquad \rho(n)=\max_k\big(\lVert\mu_n-\mu_k\rVert+\rho(k)\big),\quad \rho(\text{leaf})=2\max(s).$$

Each climb is one kernel launched over all nodes, in which every node raises its parent with `InterlockedMax` on the
float's bits. The kernel is relaunched until a counter reads zero, which is the tree's depth in rounds. The error is
projected at the sphere's nearest point:

$$P(n) = \frac{\varepsilon(n)\,f}{\max(\lVert\mu_n-e\rVert-\rho(n),\ 10^{-3})}.$$

**Proposition.** If $P(n)\le\tau<P(\mathrm{par}(n))$ selects $n$ (a root needs only the first inequality), then every
leaf is drawn exactly once. *Proof.* $\varepsilon$ does not decrease going up. The parent's sphere contains the
child's, so the distance to the sphere decreases going up. $P$ therefore does not decrease along any leaf-to-root
path. Leaves have $P=0\le\tau$, so the predicate "$P\le\tau$" holds on a prefix of the path starting at the leaf, and
exactly one node is the last of that prefix. $\square$

So the cut is a single parallel test per node (`treeErrorKeep`), with no walk up the ancestors. `--cut-budget N`
bisects $\log\tau$ over $[10^{-4},10^4]$ in 40 steps for the smallest threshold whose cut holds at most $N$ nodes.

| configuration | scene | size cut | error cut |
|---|---|---|---|
| far, 393 px, 0.25 px | six pieces | 83 k, 34.4 dB | 34 k, 34.9 dB |
| near, 426 k nodes | six pieces | 42.92 / 31.22 dB (frame / grille) | 43.61 / 34.50 dB (reference prototype 43.61 / 34.47) |

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

Two alternatives were measured. Normalising by $\sum G$ instead of $\sum G^2$ gives 0.5 for a merge as wide as one
opaque leaf. A fit over $\pm2\sigma$ instead of $\pm1.25\sigma$ gives a panel's merges 0.84, because the fringe is
where the neighbours carry the surface, and a fender comes out mottled. Leaves edge-on to the plane, such as a
grille's wires seen square on, cover nothing, so the merge goes to 0. A loader drops anything under 1/255, which would
break the tree, so a merge is written at least at

$$o_n \ge 2/255\quad(\texttt{treeCoverageFloor}).$$

The tree is rewritten in its own file order, and only the merges' opacities change (`treeCoverageToFile`), because
`lodParent`/`lodStart` are file indices that a rewrite through the Morton-ordered store would scramble. The fit reads a
Blob's axes as rows. Read as columns, they give the inverse rotation and turn every Gaussian the wrong way: a fender's
merges then come out at 0.59 of their mass where a reference fit gives 1.00.

| configuration | what | coverage | mass |
|---|---|---|---|
| fit at cut time (48 k) | grille, 393 px | 39.79 dB (from 33.87) | — ; leaves 41.59 |
| refitted tree (49.4 k) | grille, 393 px | 41.28 dB | 39.79 at 48.1 k (`--cut-coverage`) |
| 1 px, bounds, axes read as columns | six pieces near at 1920 | 40.96 frame / 32.09 grille | old tree 41.11 / 30.09 |
| axes read as rows | six pieces near at 1920 | **43.86** | 42.55 |
| axes read as rows | six pieces far at 393 | 39.95 | 39.66 |
| axes read as rows | grille near at 393 | 39.71 | **41.16** |

The grille is the exception. Neighbouring merges overlap and composite as $1-\prod$, so a little extra opacity darkens
the whole patch. The web player (Spark) draws each node's mass ($\min(\text{coverageMass},0.99)$), and the engine's
rasteriser keeps the coverage.

## 6.5 The spill metric, and why it failed for the grille

At a coarse cut the grille's holes close: in the 0.25 px cut of the 640 px car at 1920 px with back-face culling, the
near-ground region reads 2.24× the path trace, against 1.18 in the leaves. The cause is two-leaf merges of the
sub-pixel strips of `Chrome_Part_Grill`, seen face on and 3–6 px wide at $\tau=0.25$. Their cost is ~0 because a
pair's coverage, measured on its own plane, does not change. A spill is defined on the coverage fit's own grid:

$$E_n = \frac{\sum_x \max\big(a G(x) - C(x),\,0\big)}{\sum_x aG(x)}\in[0,1],$$

that is, the share of what the merge paints that its leaves do not. It matches a NumPy reference implementation to a
correlation of 0.994, and is written as `lodSpill`. The cut adds $r\,w_s\max(E-E_0,0)$ to the own error.

| 640 px car, group 0, 0.25 px | nodes | car crop | frame |
|---|---|---|---|
| $w_s=0$ | 307 665 | 16.55 dB | 22.18 dB |
| $w_s=2$ | 561 242 (+82 %) | 16.79 | 22.38 |
| $w_s=5$ | 599 983 | 16.79 | 22.38 |
| $w_s=10$ | 612 963 (+99 %) | 16.79 | 22.38 |

The spill closes the holes: from 1.90 to 1.09 of the leaves in the render, and from 4.66× to 1.0 in an image-space
hole proxy. The price is 82–99 % more nodes. Every merge spills about 0.16, and the merges that fill the holes spill
no more than a panel's merges do, so no floor $E_0$ separates the two groups. The hole is a property of the
**projection**: the merge covers nothing in 3D that its neighbours do not cover, and the gap opens only where the
strips' screen footprints fall apart. No view-independent per-node measure sees that. The default is $w_s=0$.
The 1280 px bake (§6.10) resolves the problem in the data instead: the grille's holes in the cut go from 2.24 to 1.04.

## 6.6 Back-face culling of closed shells

At small output sizes, a cut shows back faces as black. On the grille at 393 px, culling back faces raises the uncut
leaves from 38.2 to 41.6 dB. `--cut-cull-back` drops a node whose whole normal cone faces away:

$$\theta(n)<\tfrac{\pi}{2}\ \wedge\ n_n\cdot\frac{e-\mu_n}{\lVert e-\mu_n\rVert} < -\sin\theta(n).$$

The cone bounds every leaf under the node (§6.3), so the node's whole subtree faces away too. Two measurements set
where it applies. Applied to the six test pieces with the headlight's glass cover among them, it costs 6 dB at 1920:
glass is seen through, so its back faces are visible. On the whole car at 393 px with the 1 px error cut, the doors
and panels, which are two skins, show a net of dark cracks: the inner skin's merges, facing away and lit by the dark
interior, sort in front of the outer skin in patches. The net is the same with mass or coverage opacity, and
`--cut-guard 1000` does not change it. On a closed shell, what lies behind a face that looks away is the part's own
inside, so culling it is safe and removes the net. The cull is therefore applied, to leaves and merges alike, only on
**closed shells that are not glass** (`primvars:athenea:splat:closed`, transmission $\le 0.5$).

**The open-edge rule.** `mesh2splat` counts the edges not shared by exactly two triangles, $O$. Two simpler rules were
considered: $O=0$, and a share of open edges under 2 % (111 of 192 meshes closed). An open sheet's border grows as the
square root of its area, so a fine tube passes the share rule: a rim's barrel at 1 % is culled, and light shows
between the spokes. The rule used is

$$\text{closed} \iff O < 2\sqrt{E},\qquad E = 1.5\,T,$$

for a mesh of $T$ triangles. Under it, doors (0), logos (1.4) and calipers (1.5) are closed, and rims (4.0), tyres
(3.1), the hood (3.1) and the bumper (4.4) are not, which gives 102 of the Corvette's 192 meshes. With all leaves at
1920 px against the path trace, the cull lifts the whole door ~12 % evenly (door 1.01, near ground 1.02 of the path
trace), with 4.35 M → 3.70 M Gaussians and car crop 16.73 → 16.75 dB.

## 6.7 Occlusion masks

A TX transfer stores 256 shadow bits per Gaussian, one per cell of a $16\times16$ octahedral grid over the world's
sphere, set where a ray from the Gaussian reached the sky. A direction whose cells are all closed is one from which an
eye outside the asset cannot see the Gaussian. The shading bits cannot be reused for this: a merge holds its leaves'
bits by majority, and the player's codebook (`shadow_vq`, §7.3) is also a majority, so both would close directions from which
some leaf is seen. `--occlusion-mask N` (`occlusion_mask.slang`, `Occlusion.cpp`) builds a conservative mask per node.

**Masks.** A leaf's mask $M_l$ is its open cells dilated by one cell ($3\times3$), mirrored across the octahedral
square's edge at the fold. One cell is ~11°, and without the margin an occluder's edge closes directions from which a
grille's chrome is still visible. The mask is all open on glass (transmission $>0.5$, since what lies behind glass is
seen through it) and on a leaf with no bits. A merge's mask is the OR of its children's, climbed until no mask changes:

$$M_p = \bigvee_{k\in\mathrm{ch}(p)} M_k .$$

**Conservative codebook.** The masks are replaced by an index into $N$ entries of 256 bits each. Entry 0 is all open.
A node's **worth** is the number of closed cells among the *relevant* ones, the directions with $d\cdot\hat{up} >
-0.25$ (an eye down to ~15° below the horizon). The nodes in the lowest $\log_2$-worth bins, together under 2 % of the
total worth, take entry 0 and stay out of the fit. The remaining entries are seeded from fitted nodes spread through
the file. Round 1 assigns each node to the entry nearest in Hamming distance over the relevant cells. Rounds 2–8
assign a node only to an entry that **contains** its open set, choosing the one with the fewest relevant open cells.
After each round an entry becomes the OR of its members. Every entry is therefore a superset of each member's open
set, and the codebook can never close a direction a node has open. The codebook's cost is the cells it reopens
($M_{\text{entry}}\wedge\lnot M_n$) as a share of all closed cells.

**Shared by need.** One asset's trees share one codebook (`--occlusion-group`). Each unit is fitted alone at the full
size and the entries it uses are counted. The codebook is then shared out as $\text{share}_u = 16 +
\lfloor\text{spare}\cdot \text{used}_u/\sum\text{used}\rfloor$, and each unit is refitted within its share. Sharing by
fit quality and by part (`--occlusion-by-part`) were both measured and are worse:

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

(From the third row onward, the fit counts only relevant cells, so the total reopened over *all* cells rises. Those
percentages are not comparable with the first two rows.) On the close-up level, the sum of closed cells over ~150 M
nodes exceeds a 32-bit counter, so the totals are summed in 64 bits. The per-tree results are: Windshield 0.0 %
(glass, 1 entry used), Part_2 5.3 %, Hood 7.0 %, the pilot door 7.7 %, the fine groups 8.9–15.9 % and the small-panel
group pan2 23.6 %.

**The player's rule** (and `occludedFrom` in the cut). This rule applies only when the eye is outside the asset's
bounding box. Let $d=\mathrm{normalize}(e-\mu_n)$ and $q=16\,\mathrm{oct}(d)-0.5$. The four cells
$\lfloor q\rfloor+\{0,1\}^2$, clamped to $[0,15]^2$, are the bilinear $2\times2$ footprint toward the eye. The node is
culled when all four are closed in its entry. With the car as occluder (never in a partial scene), the rule removes
12 % of the near cut and 8 % of the far cut for 0.06 dB. On the whole car at 393 px, with back culling, occlusion
takes the near cut from 522 931 to 360 333 nodes.

## 6.8 Pages by error

A player cannot hold a whole tree. It needs units that it fetches whole and tests without reading their nodes.

**Alternative considered: pages by level.** Pages taken as breadth-first tops of sibling subtrees still need 114 of
the hood's 263 pages for a far cut. The bounds keep small nodes wherever a highlight or a colour turns, so the cut
spans nodes from single leaves to $2^{11}$ leaves.

**Pages by error.** A node is needed exactly when it is a root or its parent's projected error exceeds $\tau$. So the
layout follows the parent's error (`lod_pages.slang`). The **band** of a node is

$$b(n) = \mathrm{clamp}\big(\lfloor\log_2\varepsilon(\mathrm{par}(n))\rfloor+40,\ 0,\ 63\big),\qquad b(\text{root}) = 63 .$$

The sort key, in the pages' own order, is (coarsest band first, then first leaf, then larger node first):

$$\text{hi} = (63-b)\ll22 \;|\; \text{bucket}\ll19 \;|\; \text{start}\gg6,\qquad \text{lo} = (\text{start}\,\&\,63)\ll26 \;|\; (2^{26}-1-\text{leaves}).$$

A GPU radix sort orders the nodes, and page $p$ holds sorted positions $[pP,(p+1)P)$. The pages cross band
boundaries, so a coarse band of a few nodes does not become a page of its own. A parent's band is never finer than its
child's, its first leaf is never later, and when both tie the parent is the larger node. A parent is therefore never
placed after its child. (A layout that runs each band's leaves backwards puts 92 parents after their children on the
hood.) `--tree-pages P` writes, for each node, `lodPage`, and for each page:

- `lodPageInfo` = (coarsest band, nodes, first node in page order), and
- `lodPageBounds` = the box holding the **parents'** spheres (a node's need is measured from its parent) and the
  largest and smallest parent error.

The player projects the largest error from the box and loads the page when that projection exceeds $\tau$. The
bucket bits (`--pages-by-facing`: dominant normal axis and sign) make residency worse, 4.45× against 3.98× at
$\tau=2.5$, and are off.

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

`treeCulledMark` marks the nodes the culls leave out. A culled node's subtree is culled too, so a walk stops there.
That brings the nodes needed to 2.2× (a walk into culled subtrees needs ~4×), but the pages still hold 3.6–4.5×. A
streaming player can therefore keep the traversal at 12 B per node for what is needed and the payload in deflate runs
of 64 for what is drawn; in the WebGPU player of §7, near at $\tau=3$, that comes to ~60 MB and 226 k drawn at 60 fps
(emulated).

## 6.9 The close-up level

Close up, the logo and grille stay soft at the 1280 px bake. A close camera comes to about 0.5 mm per output
pixel, while the 1280 px leaves are 3–4 mm. A logo-and-grille bake from a close camera at 5120 px shows the checkered
flag, the bowtie and the grille's mesh crisp (108 527 → 404 594 Gaussians). The close-up level (car5120) re-bakes the
car in groups (Figure 9):

- **Detailed parts.** The 66 parts that the 1280 px bake gives at most 1.5 Gaussians per triangle (badges, grilles,
  lamps, calipers, mirrors, wheels, wipers, trims), at 5120 px, in 7 groups (`fine0–6`).
- **Panels.** Each part at 5120 px, or as fine as keeps it under a leaf cap of 4.5 M (a 5.01 M-leaf tree is at the
  limit of device memory when written), with the small ones grouped up to 4 M. `--max-cells 1048576` is required: at
  4736 px, Infinite_Black loses 359 628 cells past the default 262 144 per triangle, which leaves holes.

Each group is converted with every other part hidden, but the bake traces the whole stage, so a part keeps the others'
shadows. Trees are per part, so the close-up level has the same format as car1280: the coarse roots stay, and the
deeper levels are paged in only close up.

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

The 24 trees hold 148 864 157 nodes in 49 634 pages of 3000, and 87.4 GB on disk. Rebuilt from their leaves with the
second moments about each group's own centre (§6.2.1), and with a shared occlusion codebook of 8181 entries (11.9 % of
cells reopened), the car's cut at 0.25 px draws 11.7 M Gaussians instead of 16.9 M and reads 24.00 dB on the frame and
18.31 dB on the car against the path trace (23.62 and 17.91 before). Some trees keep many lone roots
(fine5: 29 213; Infinite_Black: 92 773), because merges never cross parts or guards, by design. The player's
packer (§7.5.4) adds a multi-level synthetic root above them, since a file holds about 3 000 roots.

**Memory for trees of 4–5 M leaves.** A front wheel with 4.45 M leaves and 8.88 M nodes (the tyre alone 2.56 M) builds
its tree within device memory, but writing it needs two changes.

1. `mergeAttribute` needs a weights buffer only for a rig's joints. Allocated as large as each attribute, it costs
   1.7 GB alongside each of a transfer's 48-float arrays over 8.9 M nodes. It is one word for every other attribute.
2. The write must not hold each array three times in unified memory (what was read, a copy, and the merged arrays
   started as another copy). The merged arrays start from the metadata alone, and the tree's write, the last step,
   releases each array once it is on the device.

With these changes, a 4.08 M-leaf door peaks at 10.5 GB resident and 4.90 M leaves are written. Front_Bumper_001
(5.01 M) is written with 2.23 GB of free memory at the lowest point; the 4.5 M cap for panels keeps that margin.

**Occlusion written in place.** At 87.4 GB for the 24 trees, a copy of each tree does not fit beside them on disk.
`occlusionMasksShared` writes each tree to `<tree>.occlusion.usdc` beside it and renames it over the original. The
shared codebook (8 182 entries) takes 645 s, with a peak resident set of 10.2 GB; per tree it reopens 0–23.6 % of the
closed cells (§6.7).

## 6.10 Measured results

All renders are the engine's rasteriser. "Leaves at 4×" means every leaf drawn at 4× resolution and box-reduced, the
reference for far cuts (a 1920 px bake drawn at 393 px is see-through). "PT" is the mesh path trace at 1920×1080 from
`/World/Camera`, compared on the frame and on the car's 900×420 crop.

**The 640 px car at 393 px, against the leaves at 4×.**

| configuration | cut | near nodes | near dB | far nodes | far dB |
|---|---|---|---|---|---|
| coverage trees | error 1 px, bounds | 666 585 | 33.29 | 343 541 | 39.15 |
| mass trees | error 1 px, bounds | 666 585 | 33.27 | — | — |
| coverage trees | $\tau=0.5$ | 930 119 | 33.17 | — | — |
| + back culling | closed shells by the 2 % rule | 522 931 | 33.63 | — | — |
| + occlusion | 8192 shared by group | 360 333 | 33.58 | 195 049 | 39.53 |
| both culls | $O<2\sqrt{E}$ rule (102 parts) | 378 028 | 33.53 | — | — |

The far cut (195 k) fits a budget of ~250 k nodes, the one the mobile player of §7 draws at 30 fps. The near cut is
~1.4× that.

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

The level of detail costs about 0.1 dB against the path trace (16.37 → 16.28 at 1/8 of the leaves). The bake accounts
for the rest: its resolution, its lamps and the sharpness of its reflections. The 1280 px re-bake (8.69 M leaves,
8.67 M merges, 20 723 roots, 5 789 pages, occlusion reopening 12.3 %) adds 1.3 dB on the car. Against the 640 px
0.25 px cut with the shading fixes, the 1280 px cut takes the grille's holes from 2.24 to 1.04 of the path trace, the
hood highlights from 3.2 to 2.0 px off, and the joints from 0.90 to 1.02 of the path trace.

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
- **Not ported.** The closed-shell back rule and the error cut live in `athenea-pieces`. The main engine's
  `CutSelector` is unchanged.
