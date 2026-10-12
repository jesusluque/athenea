<!-- Copyright (c) 2026 jesus luque. -->

# 6. Levels of detail: merge trees, cuts by error, pages and occlusion

A cloud baked from a mesh at 1280 px holds 8.7 million Gaussians for one car. The close-up level baked at 5120 px
holds 74.5 million. The phone's budget, set by the user, is 50–60 MB resident and about 250 k Gaussians drawn at
30 fps. Every figure between those limits depends on the level-of-detail structure. This section first describes the
engine's original level of detail (§6.1), which was built for captured scenes, and explains why it does not meet that
budget. It then covers what replaced it in the pieces work: per-part pair-merge trees (§6.2), one cut by a bounded
screen error (§6.3), the opacity a merge is given (§6.4) and a failed attempt to improve it (§6.5), two view-dependent
culls (§6.6, §6.7), pages ordered by error (§6.8), the 5120 px close-up level (§6.9) and the measurements (§6.10).
Every kernel named here is a Slang compute kernel. The CPU reads files and does bookkeeping only (§2).

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

**Why it was replaced for the phone.** The structure was mapped on 10 Oct (`PIECES-PLAN.md`), and four properties did
not fit the goal.

1. The threshold is set per instance and is not a budget. Each instance costs two blocking submits and a readback per
   frame, and no budget is shared across instances.
2. The octree cells ignore the object. A cell merges whatever falls inside it, including a badge with the panel
   beneath it and glass with paint.
3. A streamed splat costs 132 B, against the ~55 B target.
4. Only the splats are chunked. The merged levels, down to one with up to half as many cells as splats, are always
   resident, and for millions of leaves they alone exceed the budget.

## 6.2 Per-part pair-merge trees

`athenea decimate --tree` builds a binary merge tree by mutual nearest-neighbour pair merging. It is NanoGS (Xiong et
al. 2026) on the device (`lod_pairs.slang`, `lod_pairs_final.slang`, `Lod.cpp`), with guards that suit baked material
clouds.

### 6.2.1 Nodes as moments

A node is a vector of 24 floats: $W$, $W\mu$, $W(\Sigma+\mu\mu^\top)$ (6), $W\cdot\mathrm{sRGB}(c)$, $W n$, $W\alpha$,
$W\cdot$metallic, $W\cdot$transmission, $W\cdot$specular, the key, the record count, the weighted thickness ratio and
the node's index in the tree. Here $w = o\cdot\max(s_xs_y, s_ys_z, s_xs_z)$ (`discArea`), and $\alpha$ is the GGX
roughness squared. A merge sums the two vectors. Because sums are associative, a node's moments are those of all its
leaves whatever the merge order. Every other attribute (transfer, shadow bits, material, joints) is merged once at the
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
to cut to: one headlight cover went from 76 486 roots to 123 (3d7ee66).

**Layout.** The tree is written as a ParticleField. The leaves come first, in depth-first order, so that every node is
a contiguous run of leaves. The merges follow in the order they were made. Each node carries `lodParent` (a root is
its own parent), `lodStart` (its first leaf), `lodSize` (its number of leaves) and `lodCost`.

### 6.2.5 Many parts: `--passes`, `--merge-glass`, `--object-groups`

On the whole Corvette (5.38 M leaves), building the tree in one pass worked, but writing its attributes ran out of
device memory. `--object-groups K --object-group i` keeps the Gaussians whose `cryptoObject` mod $K$ is $i$. Merges
never cross parts, so each group's tree is exactly the subtree the whole asset would hold for those parts. Each pass
merges at most half of what is left, and a group of many small parts merges few pairs at once. The default cap of 60
passes left a group with 116 549 roots out of 323 613 leaves. At `--passes 1000` it reached 1 298 roots in 310 passes
(94fa9f1). `--tree-report` showed two causes of the remaining roots: solid glass never merged (roots equal leaves), and
the tyres kept merging a trickle per pass until the cap.

| group (640 px car) | roots before | with `--merge-glass --passes 4000` |
|---|---|---|
| group 2 | 190 490 | 4 237 |
| group 5 | 36 508 | 817 |
| whole car (8 groups) | — | 17 133 roots, 5.38 M leaves, 3 582 pages |

## 6.3 The cut by error, with bounds

The first cut (`--cut-pixels`, 655c027) kept a node when its on-screen width $2\max(s)f/d$ was at most $\tau$ and no
ancestor's was. Merges past the guards passed only at $0.25\tau$. That cut is still available, but an error cut
replaced it (2bb1b80, e13b08b; research 100, after Nanite).

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

| test | scene | size cut | error cut |
|---|---|---|---|
| T15 | six pieces, far, 393 px, 0.25 px | 83 k, 34.4 dB | 34 k, 34.9 dB |
| T18 | six pieces, near, research's 426 k | 42.92 / 31.22 dB (frame / grille) | 43.61 / 34.50 dB (prototype 43.61 / 34.47) |

## 6.4 The opacity of a merge: least-squares coverage

Mass over the moment ellipse counts the leaves' overlapping tails twice. Research measured the effect at 27 % at the
median and 2.3× at p10. On the grille at 393 px, merges drew darker than their leaves. `--cut-coverage` (0ab577e) and
then `--tree-coverage` (73c0ec2, f980e42) set every merge's opacity from what its leaves actually cover, as follows.

1. The node's plane is spanned by its two widest axes $u,v$, with scales $s_u,s_v$. An $8\times8$ grid covers the
   core $[-1.25,1.25]^2$ in units of $(s_u,s_v)$.
2. Each leaf $l$ under the node (its run, up to 4096 leaves when writing the tree and 2048 in a cut) is projected onto
   the plane. Its in-plane covariance is $(u^\top\Sigma_lu,\ u^\top\Sigma_lv,\ v^\top\Sigma_lv)$ and its centre
   offset is $m_l$. A longer run keeps its mass-based opacity. The leaves are composited as the rasteriser would:
   $C(x)=1-\prod_l\big(1-\min(o_l,0.99)\,G_l(x)\big)$.
3. The node's own footprint on the grid is $G(x)=\exp(-\tfrac12\lVert x\rVert^2)$. The opacity is the least-squares
   fit of $aG$ to $C$:

$$a^\star = \frac{\sum_x G(x)\,C(x)}{\sum_x G(x)^2},\qquad o_n = \mathrm{clamp}(a^\star,0,0.99).$$

Two choices were measured before this form was kept. Normalising by $\sum G$ instead of $\sum G^2$ gave 0.5 for a
merge as wide as one opaque leaf. A fit over $\pm2\sigma$ instead of $\pm1.25\sigma$ gave a panel's merges 0.84,
because the fringe is where the neighbours carry the surface, and a fender came out mottled. Leaves edge-on to the
plane, such as a grille's wires seen square on, cover nothing, so the merge goes to 0. A loader drops anything under
1/255, which would break the tree, so a merge is written at least at

$$o_n \ge 2/255\quad(\texttt{treeCoverageFloor}).$$

The tree is rewritten in its own file order. Only the merges' opacities change (`treeCoverageToFile`), because
`lodParent`/`lodStart` are file indices and an earlier rewrite through the Morton-ordered store had scrambled them.
One bug was found by research: the fit read a Blob's axes as columns, the inverse rotation, and so turned every
Gaussian the wrong way. A fender's merges came out at 0.59 of their mass where research's fit gave 1.00 (4dce865).

| test | what | coverage | mass |
|---|---|---|---|
| T20 | grille, 393 px, cut at fit time (48 k) | 39.79 dB (from 33.87) | — ; leaves 41.59 |
| T21 | grille, 393 px, refitted tree (49.4 k) | 41.28 dB | 39.79 at 48.1 k (`--cut-coverage`) |
| T24f | six pieces near at 1920, 1 px, bounds | 40.96 frame / 32.09 grille | old tree 41.11 / 30.09 |
| T47 | six pieces near at 1920 (after the axes fix) | **43.86** | 42.55 |
| T47 | six pieces far at 393 | 39.95 | 39.66 |
| T47 | grille near at 393 | 39.71 | **41.16** |

The grille is the exception. Neighbouring merges overlap and composite as $1-\prod$, so a little extra opacity darkens
the whole patch. The web player (Spark) draws each node's mass ($\min(\text{coverageMass},0.99)$), and our raster keeps
the coverage.

## 6.5 The spill metric, and why it failed for the grille

At a coarse cut the grille's holes closed: in T66 the near-ground region read 2.24× the path trace in the 0.25 px cut,
against 1.18 in the leaves. Research traced this to two-leaf merges of the sub-pixel strips of `Chrome_Part_Grill`,
seen face on and 3–6 px wide at $\tau=0.25$. Their cost is ~0 because a pair's coverage, measured on its own plane,
does not change. A spill was defined on the coverage fit's own grid (1bb79f4):

$$E_n = \frac{\sum_x \max\big(a G(x) - C(x),\,0\big)}{\sum_x aG(x)}\in[0,1],$$

that is, the share of what the merge paints that its leaves do not. It matched research's numpy to a correlation of
0.994, and is written as `lodSpill`. The cut adds $r\,w_s\max(E-E_0,0)$ to the own error.

| T67, group 0, 0.25 px | nodes | car crop | frame |
|---|---|---|---|
| $w_s=0$ | 307 665 | 16.55 dB | 22.18 dB |
| $w_s=2$ | 561 242 (+82 %) | 16.79 | 22.38 |
| $w_s=5$ | 599 983 | 16.79 | 22.38 |
| $w_s=10$ | 612 963 (+99 %) | 16.79 | 22.38 |

The spill did close the holes: from 1.90 to 1.09 of the leaves in the render, and from 4.66× to 1.0 in research's
proxy. The price was 82–99 % more nodes. Every merge spills about 0.16, and the merges that fill the holes spill no
more than a panel's merges do, so no floor $E_0$ separates the two groups. The hole is a property of the
**projection**: the merge covers nothing in 3D that its neighbours do not cover, and the gap opens only where the
strips' screen footprints fall apart. No view-independent per-node measure sees that. The default is $w_s=0$.
The 1280 px bake (§6.10) fixed the problem in the data instead: the grille's holes in the cut went from 2.24 to 1.04.

## 6.6 Back-face culling of closed shells

At the phone's size, a cut showed back faces as black. On the grille at 393 px, culling back faces raised the uncut
leaves from 38.2 to 41.6 dB (research). `--cut-cull-back` drops a node whose whole normal cone faces away:

$$\theta(n)<\tfrac{\pi}{2}\ \wedge\ n_n\cdot\frac{e-\mu_n}{\lVert e-\mu_n\rVert} < -\sin\theta(n).$$

The cone bounds every leaf under the node (§6.3), so the node's whole subtree faces away too. Two observations set
where it applies. Applied to the six test pieces with the headlight's glass cover among them, it cost 6 dB at 1920:
glass is seen through, so its back faces are visible. On the whole car at 393 px (T50), the doors and panels, which
are two skins, showed a net of dark cracks: the inner skin's merges, facing away and lit by the dark interior, sorted
in front of the outer skin in patches (T51: the same with or without coverage; `--cut-guard 1000` changed nothing). On
a closed shell, what lies behind a face that looks away is the part's own inside, so culling it is safe and removes
the net (T56). The cull is therefore applied, to leaves and merges alike, only on **closed shells that are not
glass** (`primvars:athenea:splat:closed`, transmission $\le 0.5$).

**The open-edge rule.** `mesh2splat` counts the edges not shared by exactly two triangles, $O$. It first marked a mesh
closed when $O=0$ (40a7927) and then under 2 % of its edges (2b05ebb, 111 of 192 meshes). An open sheet's border grows
as the square root of its area, so a fine tube passed the share rule: a rim's barrel at 1 % was culled, and light
showed between the spokes on the phone. The rule kept is

$$\text{closed} \iff O < 2\sqrt{E},\qquad E = 1.5\,T,$$

for a mesh of $T$ triangles (f52b6b4). Under it, doors (0), logos (1.4) and calipers (1.5) are closed, and rims (4.0),
tyres (3.1), the hood (3.1) and the bumper (4.4) are not, which gives 102 of the Corvette's 192 meshes. T66 (all
leaves, 1920 px, against the path trace): the cull lifts the whole door ~12 % evenly (door 1.01, near ground 1.02 of
the path trace), with 4.35 M → 3.70 M Gaussians and car crop 16.73 → 16.75 dB.

## 6.7 Occlusion masks

A TX transfer stores 256 shadow bits per Gaussian, one per cell of a $16\times16$ octahedral grid over the world's
sphere, set where a ray from the Gaussian reached the sky. A direction whose cells are all closed is one from which an
eye outside the asset cannot see the Gaussian. The shading bits cannot be reused for this: a merge holds its leaves'
bits by majority, and the phone's codebook (`shadow_vq`) is also a majority, so both would close directions from which
some leaf is seen. `--occlusion-mask N` (1000bdf, `occlusion_mask.slang`, `Occlusion.cpp`) builds a conservative mask
per node.

**Masks.** A leaf's mask $M_l$ is its open cells dilated by one cell ($3\times3$), mirrored across the octahedral
square's edge at the fold. One cell is ~11°, and without the margin an occluder's edge closed directions from which a
grille's chrome was still visible. The mask is all open on glass (transmission $>0.5$, since what lies behind glass is
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

**Shared by need.** One asset's trees share one codebook (`--occlusion-group`, 5be8710). Each unit is fitted alone at
the full size and the entries it uses are counted. The codebook is then shared out as $\text{share}_u = 16 +
\lfloor\text{spare}\cdot \text{used}_u/\sum\text{used}\rfloor$, and each unit is refitted within its share (1103f9f).
Sharing by fit quality (T41) and by part (`--occlusion-by-part`, 0fde773) were both worse:

| test | trees | codebook | reopened | culling vs ideal |
|---|---|---|---|---|
| T27 | 6 pieces, one book each | 256 | 3.2–10 % | — |
| T30 | 6 pieces, one fit over all | 4096 / 1024 | 3.2 % / 6.5 % | — |
| T32 | + worth, relevant cells only | 4096 / 1024 | 21.9 % / 24.7 % | — |
| T39 | worth = closed cells, not leaves | 4096 | 14.7 % | — |
| T40 | shared by need | 4096 | 7.2 % (grille 2.8 %, 1 249 entries) | 86 % near / 82 % far |
| car 640 | 8 groups by need | 4096 | 16.2 % | — |
| car 640 | 8 groups by need | 8192 | 11.9 % | 84 % |
| car 640 | by part (192) | 4096 / 8192 | — | 66 % / 74 % |
| car 1280 | 8 groups by need | 8187 | 12.3 % | — |
| car 5120 | 24 trees by need | 8182 | 0–23.6 % per tree | — |

(From T32 onward, the fit counts only relevant cells, so the total reopened over *all* cells rises. Those percentages
are not comparable with T27–T30.) On the close-up level, the total reported "132.4 %" because the closed-cell total
over ~150 M nodes wrapped a 32-bit counter. Each tree's figure and the codebook itself were correct, and the totals
are now summed in 64 bits (4a8f8c2). The per-tree results were: Windshield 0.0 % (glass, 1 entry used), Part_2 5.3 %,
Hood 7.0 %, the pilot door 7.7 %, the fine groups 8.9–15.9 % and the small-panel group pan2 23.6 %.

**The player's rule** (and `occludedFrom` in the cut). This rule applies only when the eye is outside the asset's
bounding box. Let $d=\mathrm{normalize}(e-\mu_n)$ and $q=16\,\mathrm{oct}(d)-0.5$. The four cells
$\lfloor q\rfloor+\{0,1\}^2$, clamped to $[0,15]^2$, are the bilinear $2\times2$ footprint toward the eye. The node is
culled when all four are closed in its entry. Research measured −12 % near and −8 % far on the cut for 0.06 dB, with
the car as occluder (never in a partial scene). On the whole car at 393 px, with back culling, occlusion took the near
cut from 522 931 to 360 333 nodes.

## 6.8 Pages by error

A player cannot hold a whole tree. It needs units that it fetches whole and tests without reading their nodes.

**By level fails.** Pages taken as breadth-first tops of sibling subtrees still needed 114 of the hood's 263 pages for
a far cut. The bounds keep small nodes wherever a highlight or a colour turns, so the cut spans nodes from single
leaves to $2^{11}$ leaves (e562899).

**By error works.** A node is needed exactly when it is a root or its parent's projected error exceeds $\tau$. So the
layout follows the parent's error (`lod_pages.slang`). The **band** of a node is

$$b(n) = \mathrm{clamp}\big(\lfloor\log_2\varepsilon(\mathrm{par}(n))\rfloor+40,\ 0,\ 63\big),\qquad b(\text{root}) = 63 .$$

The sort key, in the pages' own order, is (coarsest band first, then first leaf, then larger node first):

$$\text{hi} = (63-b)\ll22 \;|\; \text{bucket}\ll19 \;|\; \text{start}\gg6,\qquad \text{lo} = (\text{start}\,\&\,63)\ll26 \;|\; (2^{26}-1-\text{leaves}).$$

A GPU radix sort orders the nodes, and page $p$ holds sorted positions $[pP,(p+1)P)$. The pages cross band
boundaries, so a coarse band of a few nodes does not become a page of its own. A parent's band is never finer than its
child's, its first leaf is never later, and when both tie the parent is the larger node. A parent is therefore never
placed after its child. An earlier layout ran each band's leaves backwards and put 92 parents after their children on
the hood (e2c2046). `--tree-pages P` writes, for each node, `lodPage`, and for each page:

- `lodPageInfo` = (coarsest band, nodes, first node in page order), and
- `lodPageBounds` = the box holding the **parents'** spheres (a node's need is measured from its parent) and the
  largest and smallest parent error.

The player projects the largest error from the box and loads the page when that projection exceeds $\tau$. The
bucket bits (`--pages-by-facing`: dominant normal axis and sign) made residency worse, 4.45× against 3.98× at
$\tau=2.5$, and are off.

**Residency.** A walk needs the cut and all its ancestors, about 2× the cut. Pages hold about 4× the cut whatever
their size or order, because needed and culled nodes share every page while the culling depends on the view.

| test | scene | $\tau$ | cut | nodes needed | pages holding (× cut) |
|---|---|---|---|---|---|
| T38/T42 | six pieces, near 1920, P 3000 | 1 | 244 114 | — | 2.0–2.6× |
| T38/T42 | six pieces, far 393 | — | — | — | 2.2–4.9× (hood: 2 pages) |
| T60 | car 640, near 393, P 3000 | 2 | 293 629 | 647 707 (2.20×) | 1 170 000 (3.98×) |
| T60 | | 2.5 | 263 918 | 580 997 (2.20×) | 1 050 000 (3.97×) |
| T60 | | 3 | 198 702 | 445 195 (2.24×) | 957 000 (4.81×) |
| T60 | | 6.75 | 92 987 | 214 827 (2.31×) | 423 000 (4.54×) |
| T59 | same, $\tau=2$, P 1000 / 500 | 2 | 293 629 | — | 3.91× / 3.87× |
| T61 | same, $\tau=2.5$, P 1000 / 500 | 2.5 | 263 918 | — | 3.77× / 3.60× |

`treeCulledMark` marks the nodes the culls leave out. A culled node's subtree is culled too, so a walk stops there.
That brings the nodes needed to 2.2× (a walk into culled subtrees needed ~4×), but the pages still hold 3.6–4.5×
(a16da43). The phone's player (v5, sparkwebGPU) therefore keeps the traversal at 12 B per node for what is needed and
the payload in deflate runs of 64 for what is drawn. Near, at $\tau=3$, that came to ~60 MB and 226 k drawn at 60 fps
(emulated).

## 6.9 The close-up level

On the iPhone close up, the logo and grille stayed soft. The camera came to about 0.5 mm per phone pixel, while the
1280 px leaves were 3–4 mm. A logo-and-grille test baked from a close camera at 5120 px showed the checkered flag, the
bowtie and the grille's mesh crisp (108 527 → 404 594 Gaussians). The close-up level (car5120) re-bakes the car in
groups:

- **Detailed parts.** The 66 parts that the 1280 px bake gave at most 1.5 Gaussians per triangle (badges, grilles,
  lamps, calipers, mirrors, wheels, wipers, trims), at 5120 px, in 7 groups (`fine0–6`).
- **Panels.** Each part at 5120 px, or as fine as keeps it under a leaf cap (5 M, then 4.5 M after a 5.01 M tree did
  not fit), with the small ones grouped up to 4 M. `--max-cells 1048576` was needed: at 4736 px, Infinite_Black lost
  359 628 cells past the default 262 144 per triangle (holes) and was re-baked.

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

The 24 trees hold 148 864 157 nodes in 49 634 pages of 3000, and 87.4 GB on disk. Some trees keep many lone roots
(fine5: 29 213; Infinite_Black: 92 773), because merges never cross parts or guards, by design. The phone's
packer adds a multi-level synthetic root above them, since a file holds about 3 000 roots.

**Memory for trees of 4–5 M leaves** (5ed0b6a). The first group, a front wheel with 4.45 M leaves and 8.88 M nodes
(the tyre alone 2.56 M), built its tree and then ran out of device memory writing it. Two causes were found.

1. `mergeAttribute` allocated a weights buffer as large as each attribute, though only a rig's joints write it. That
   was 1.7 GB alongside each of a transfer's 48-float arrays over 8.9 M nodes. It is now one word elsewhere.
2. The write held each array three times in unified memory: what was read, a copy, and the merged arrays started as
   another copy. The merged arrays now start from the metadata alone, and the tree's write, the last step, releases
   each array once it is on the device.

After the fix, a 4.08 M-leaf door peaked at 10.5 GB resident. 4.90 M leaves passed. The first attempt at
Front_Bumper_001 (5.01 M) ran out of memory, which led to the 4.5 M cap for later panels. A later run wrote it, with
2.23 GB as the lowest free memory under `memguard`.

**Occlusion written in place.** The 24 trees take 87.4 GB (about 86 GB was the estimate before the last tree), and a copy of each would not fit
beside them on disk.
`occlusionMasksShared` writes each tree to `<tree>.occlusion.usdc` beside it and renames it over the original. The
shared codebook (8 182 entries) took 645 s, with a peak resident set of 10.2 GB; per tree it reopens 0–23.6 % of the closed cells (§6.7).

## 6.10 Measured results

All renders are our rasteriser. "Leaves at 4×" means every leaf drawn at 4× resolution and box-reduced, the reference
for far cuts (a 1920 px bake drawn at 393 px is see-through). "PT" is the mesh path trace at 1920×1080 from
`/World/Camera`, compared on the frame and on the car's 900×420 crop.

**The 640 px car at the phone's size (393 px), against the leaves at 4×.**

| test | cut | near nodes | near dB | far nodes | far dB |
|---|---|---|---|---|---|
| T50 | error 1 px, bounds | 666 585 | 33.29 | 343 541 | 39.15 |
| T51 | same, trees before coverage (mass) | 666 585 | 33.27 | — | — |
| T52 | $\tau=0.5$ | 930 119 | 33.17 | — | — |
| T56 | + back culling of closed shells (2 % rule) | 522 931 | 33.63 | — | — |
| T57 | + occlusion (8192 shared by group) | 360 333 | 33.58 | 195 049 | 39.53 |
| T58 | both culls, $O<2\sqrt{E}$ rule (102 parts) | 378 028 | 33.53 | — | — |

The far cut (195 k) fits the phone's ~250 k at 30 fps. The near cut is ~1.4× that.

**Against the path trace (1920 px).**

| test | what | nodes | car crop | frame |
|---|---|---|---|---|
| T63b | 640 px, all leaves | 5.38 M | 16.37 | 21.92 |
| T63b | error 1 px, both culls | 675 760 | 16.28 (31.91 vs leaves) | 21.80 |
| T63c | error 0.25 px, back culling | 2 313 541 | 16.29 (34.01 vs leaves) | 21.81 |
| T64 | leaves, unwidened sharp lobes | 5.38 M | 16.58 | 22.15 |
| T65 cut_b | 0.25 px + shading fixes | — | 16.55 | 22.18 |
| T66 | leaves (0.01 px), no cull / cull | 4.35 M / 3.70 M | 16.73 / 16.75 | 22.40 / 22.43 |
| T67 | group 0 with spill $w_s=2$ | 561 242 (g0) | 16.79 | 22.38 |
| T71 | **1280 px bake**, 0.25 px, back culling | 4 998 976 | **17.88** | **23.90** |
| T75 | T71 + the tube/fold split (§4.9.4) | — | 17.88 | 23.91 |

The level of detail costs about 0.1 dB against the path trace (T63b: 16.37 → 16.28 at 1/8 of the leaves). The bake
accounts for the rest: its resolution, its lamps and the sharpness of its reflections. The 1280 px re-bake (8.69 M
leaves, 8.67 M merges, 20 723 roots, 5 789 pages, occlusion reopening 12.3 %) bought 1.3 dB on the car. Research,
comparing T71 with cut_b: the grille's holes in the cut went from 2.24 to 1.04, hood highlights from 3.2 to 2.0 px
off, and joints from 0.90 to 1.02 of the path trace.

<!-- Figures: /Users/muriel/luc/athenea-renders/pieces/car/t50/cut_Camera.png (T50 near, the crack net),
/Users/muriel/luc/athenea-renders/pieces/car/t50/cull_Camera.png (with back culling),
/Users/muriel/luc/athenea-renders/pieces/close/r_1280.png and r_5120.png (logo test, 1280 vs 5120 px),
/Users/muriel/luc/athenea-renders/pieces/close/close_mos.png -->

## 6.11 Limits and open items

- **Projection-only holes.** A coarse cut of sub-pixel strips (the grille) fills holes that no per-node, view-free
  measure can see (§6.5). This is resolved in practice by finer leaves, not by the cut.
- **Occlusion weighting.** The codebook is shared by entries used, not by how often a view culls a node. Weighting by
  an orbit of views in the bake should lift culling past 84 % of the ideal.
- **Residency at ~4×.** It is bound by pages that mix needed and view-culled nodes. The phone splits traversal from
  payload rather than shrinking pages.
- **Glass merges.** Every thin-sheet Gaussian is lifted to the 0.1 sheet floor, so merging fewer, larger sheets varies
  the transmission cell to cell. Spark declares sheet merges at 4× their size in the interim, and a floor per pixel
  rather than per Gaussian is pending.
- **Far, thin trims** still become fat dashes in every cut. The bounds miss this error.
- **Not ported.** The closed-shell back rule and the error cut live in `athenea-pieces`. The main engine's
  `CutSelector` is unchanged.
