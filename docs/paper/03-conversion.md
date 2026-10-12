<!-- Copyright (c) 2026 jesus luque. -->

# 3. From USD meshes to Gaussians (mesh2splat)

The conversion turns a USD stage of polygonal meshes and their materials into a cloud of anisotropic 3D Gaussians
that the rest of the system draws, bakes, decimates and streams. It is invoked as `athenea mesh2splat STAGE`, and it
is the core of the product: everything the phone eventually plays back was a triangle of a mesh first. This section
describes how the stage is read, how large a Gaussian's cell is, how each Gaussian is placed, shaped and coloured,
how the conversion handles the places where a surface stops or folds, how it captures a layered physically based
material, how it treats glass, and what was measured on the Chevrolet Corvette C7 asset that drives most of the
project. The path-traced bake that follows emission (the *transfer*) is the subject of Section 4 and is
touched on here only where it constrains the conversion.

Numbers in this section come from `docs/decisions.md` and `docs/pieces.md` of the repository, from the conversion
logs under `~/luc/athenea-renders/`, and from the source (`plugins/mesh2splat/mesh2splat.slang`,
`apps/athenea/src/CmdMesh2Splat.cpp`, `shaders/athenea/usd/mesh2splat_*.slang`). All measurements were taken on an
Apple M5 Pro (Metal) unless said otherwise.

## 3.1 Origin, and where the conversion runs

The emission algorithm is that of Electronic Arts' *mesh2splat* (BSD-3): every triangle is projected onto the axis
plane its normal points along most, the projection is rasterised at a fixed resolution, and each covered texel
becomes one Gaussian lying in the triangle's plane, as wide as a texel measured in the world. The original is an
OpenGL geometry and fragment shader pair; here it is one compute kernel in which each thread owns a triangle and
walks the cells of its projected bounding box.

The kernel is not a module of the engine but an AOFX effect, the bundle `plugins/mesh2splat`, loaded through the
same host registry every program shares (the AOFX ABI, Section 2.14). The conversion is a stateless kernel over
triangles, which is exactly the shape of an AOFX effect, and making it one keeps a converter out of the renderer's
modules and lets the same bundle run in other AOFX hosts. The cost is a boundary: an effect is handed pictures, so a
mesh travels as a *picture of numbers*, six `float4` per triangle — $(\mathbf p, u)$ and $(\mathbf n, v)$ for each
corner, in world space (`mesh_pack.slang`) — and every texture a material names travels as rows of linear `float4`
sampled out of the engine's texture table (`texture_rows.slang`). Neither is a copy: the pictures live in the gpe
image storage that adopts the engine's Metal device, and `Context::renderView` gives a view of the same bytes, so the
packing kernel writes where the effect reads. The records the effect writes are gathered into the cloud's device
buffers (`mesh2splat_gather`), the cloud's bounds are folded on the device (`mesh2splat_span`), and the bake reads
them in place. Only counters and the final USD arrays cross to the processor.

The command is therefore a pipeline of device passes orchestrated by the host:

1. read the stage (`usd::MeshStage`) and split meshes into *pieces*;
2. pack each piece into a mesh picture, fold boxes, choose each piece's cell (`mesh2splat_cells`);
3. mark hard edges and closed shells per mesh (`mesh2splat_edges`), classify glass (`mesh2splat_sheet`);
4. run the Mesh2Splat effect per piece — count, scan, emit — into records;
5. optionally bake (radiance, or a transfer) at every Gaussian with the path tracer;
6. write a `ParticleField3DGaussianSplat` stage (or a `.athc`) atomically.

## 3.2 Reading the stage

`usd::MeshStage` walks the stage with `UsdShade` directly rather than through Hydra: the conversion wants no render
index, and the material side is built on the stage's own bindings.

**Selection.** `--prim P` restricts conversion to meshes at or under `P`; `--hide P` (repeatable) authors session
invisibility on `P` and everything below it before the read. A mesh that computes invisible at the read's time code
is skipped, so the conversion converts what the renderer draws. `--hide` affects the conversion only: the bake
traces the whole stage, so a part converted with every other part hidden still receives the others' shadows and
inter-reflections. This is what allows the Corvette to be converted in groups (Section 3.11).

**Instancing.** A `PointInstancer` prototype's mesh is emitted once per instance of the innermost instancer that
holds it, at (mesh relative to its prototype root) × (instance transform, prototype root included) × (instancer to
world); the built mesh buffers are shared between copies. Before this, the chess set's sixteen pawns became two.
Nested instancers are expanded at the innermost level only.

**Time and pose.** `--time T` sets the time code at which positions, transforms and primvars (including their
indices) are read and at which the bake traces. A skinned mesh's `points` do not animate in USD, so the stage is
first posed with `UsdSkelBakeSkinning` into the session layer over $[T, T]$, which costs one pose and leaves the file
untouched; a stage without a `SkelRoot` is unchanged and converts byte-identically. With `--skinned` the Gaussians
are built in the bind pose and each keeps four joints and their weight gradients; that path is described in
Section 5.

**Pieces.** A mesh whose `materialBind` GeomSubsets bind different materials is converted as several *pieces*: each
subset's triangles, and those no subset claims. A piece's triangles are listed on the device in the mesh's order
(flag, prefix sum, scatter) and packed as if they were a mesh, so the effect is untouched. A test mesh with two faces
bound red and blue by subsets and the mesh bound green converts to 272 red and 272 blue Gaussians and none green;
before the change it was all green.

**Materials in their own words.** The bound surface — MaterialX `standard_surface`, OpenPBR `open_pbr_surface`,
`gltf_pbr` or `UsdPreviewSurface` — is read with its own input names and its own defaults (standard_surface 0.8 grey
at roughness 0.2, OpenPBR roughness 0.3, glTF a fully rough metal, UsdPreviewSurface 0.18 at 0.5), following
connections through node graphs and interface inputs to the image node that produces a value. A value computed by a
graph (a mix, a noise) has no answer without shading a point, so the default stands and a warning is printed; a TX
bake later reads such fields back from the compiled material on the device. A texture's colour space is taken from
the file input's `colorSpace` metadata (or `sourceColorSpace`), and maps are capped at `--texture-size` (1024 by
default): a map is a `float4` picture on the device, so a 4k map would occupy 268 MB.

## 3.3 The size of a cell

Everything downstream scales with the cell $h$, the world length one Gaussian stands for. The conversion offers four
ways to choose it, in increasing order of preference.

**Per model** (`--density per-model`, the default when no camera is given). $h = R / N$, where $N$ is
`--resolution` (512 by default) and $R$ the wider of the projection plane's two ranges of the model's box — the
original's `orthogonalUvs`. Two defects followed: a 16-unit floor diluted the car standing on it, and a panel
facing forward walked a cell 2.5 times finer than the hood, because its projection plane is the car's short side.

**Per mesh** (`--density per-mesh`). $h = L_m / N$ with $L_m$ the longest side of the mesh's own box, the same for
every triangle of it (`cellByLongest`), clamped to $[h_{\min}, h_{\max}]$ given by `--cell-min`/`--cell-max`; left at
0 they derive from the model's cell $h_M$: $h_{\max} = h_M$, $h_{\min} = h_M/8$, a density ratio of 64 between the
finest and coarsest part. Measured on `MustangHero.usda` (car on a 16-unit floor) at `--resolution 512`:

| mode | total | bumper `Object_199` | hood `Kapoot` | floor |
|---|---|---|---|---|
| per-model | 375 199 | 762 (cell 0.0313) | 2 326 | 262 144 |
| per-mesh, defaults | 934 338 | 3 082 (cell 0.0039) | 20 279 | 262 144 |
| per-mesh, `--cell-min` = `--cell-max` = 0.03125 | 375 157 | 764 | 2 339 | 262 144 |

The control row reproduces per-model to 0.01 %; the remaining difference is the orientation effect alone.

**From a camera** (`--cell-from-camera CAM --camera-pixels P`, after Mesh2GS, arXiv 2606.21898). Each piece's cell
is what one pixel of camera `CAM`, `P` pixels across, covers where the piece's mesh box is nearest to it:

$$
h = \frac{z\,(A/P)}{f}, \qquad z = \max\bigl(\lVert \operatorname{clamp}(\mathbf e, \mathbf b_{\min}, \mathbf b_{\max}) - \mathbf e \rVert,\; z_{\text{near}}\bigr),
$$

with $A$ the camera's horizontal aperture, $f$ its focal length, $\mathbf e$ the eye, and the result bounded only by
the user's `--cell-min`/`--cell-max`. Mesh2GS takes $z_{\min}$ over the box inside the frustum and over the camera's
time samples; the Euclidean distance at `--time` is the same for a box in view and finer, never coarser, for one
beside the frustum. On two equal cards three and seven units from a 64-pixel camera, the near card received 930
Gaussians and the far one 169: a ratio of 5.5 against $(7/3)^2 = 5.4$ expected. This is the mode used for every
production bake of the Corvette (`--cell-from-camera /World/Camera --camera-pixels` 640, 1280 or up to 5120).

**The budget.** `--max-splats` (2 M by default) is shared over the whole stage. Every piece is first counted (a run
of the effect with room for one Gaussian, whose count and scan cover everything a run would write); where the total
$W$ exceeds the budget $B$, piece $i$ wanting $w_i$ gets $s_i = \max(1, B w_i / W)$ and walks a cell
$\sqrt{w_i / s_i}$ times coarser. A cell $k$ times coarser covers the surface with $k^2$ fewer Gaussians, so
proportional shares and uniform coarsening are the same thing. On two cards wanting about 1 300 under a budget of
800 the shares came out 356 and 361; spent in mesh order, the second card held about 145.

**Per-triangle and per-run limits.** A triangle walks at most `--max-cells` cells of its projected bounding box
($2^{18}$ by default); cells beyond are counted and reported as a warning. At 4736 px the Corvette's
`Infinite_Black` lost 359 628 cells this way (visible holes) and was re-baked with `--max-cells 1048576`. A run's
output picture is capped at 2 M Gaussians (`kRunCeiling`, 192 MB); a piece wanting more is converted in slices — the
effect reports the first triangle the ceiling cut into, and the host reruns from there with `firstTriangle` — giving
the same array a single run would have written.

## 3.4 One Gaussian per cell

**The projection.** For a triangle $(\mathbf a, \mathbf b, \mathbf c)$ with geometric normal $\hat{\mathbf n}_g$,
the two axes $(i, j)$ along which $\hat{\mathbf n}_g$ points least define the projection; a point maps to
$\mathbf o(\mathbf x) = (x_i - b_{\min,i},\, x_j - b_{\min,j}) / R$, with $R$ the range in use (the model's, the
mesh's longest side, or the camera-derived box). A cell is $1/N$ of that unit square, i.e. $h = R/N$ in the world,
bounded as in Section 3.3. A Gaussian is emitted at every cell whose centre falls inside the projected triangle,
located by its barycentric coordinates $\boldsymbol\lambda$, at the world point
$\boldsymbol\mu = \lambda_0 \mathbf a + \lambda_1 \mathbf b + \lambda_2 \mathbf c$.

**The sizes.** With $E = [\mathbf b - \mathbf a,\; \mathbf c - \mathbf a]$ and $O = [\mathbf o_b - \mathbf o_a,\;
\mathbf o_c - \mathbf o_a]$, the Jacobian of the projection's inverse is $J = E\,O^{-1} = [\mathbf j_u, \mathbf j_v]$
(guarded at $|\det O| > 10^{-24}$). The two in-plane standard deviations and the third are

$$
s_u = \min\!\left(\lVert \mathbf j_u \rVert\,\sigma\,\tfrac{1}{N},\; \ell_{\max}\right), \qquad
s_v = \min\!\left(\lVert \mathbf j_v \rVert\,\sigma\,\tfrac{1}{N},\; \ell_{\max}\right), \qquad
s_n = \varphi\,\min(s_u, s_v),
$$

where $\sigma$ is `--sigma`, $\ell_{\max}$ the triangle's longest edge and $\varphi$ `--flatness`. Three details were
read from the original source rather than guessed: the Jacobian is over the triplanar coordinates, not over texture
coordinates; the $1/N$ factor is applied by the original's exporter, not by a shader (without it every Gaussian is
the size of the model); and the frame is not the Jacobian's columns but the triangle's longest edge
$\hat{\mathbf t}$, the interpolated shading normal $\hat{\mathbf n} = \operatorname{normalize}(\sum_k \lambda_k
\mathbf n_k)$, and their cross product. The covariance is therefore

$$
\Sigma = R_{\text{frame}}\,\operatorname{diag}(s_u^2, s_v^2, s_n^2)\,R_{\text{frame}}^{\!\top}, \qquad
R_{\text{frame}} = [\hat{\mathbf t}_\perp,\; \hat{\mathbf n}\times\hat{\mathbf t}_\perp,\; \hat{\mathbf n}],
$$

with $\hat{\mathbf t}_\perp$ the edge made orthogonal to $\hat{\mathbf n}$. The bound $\ell_{\max}$ was added after
1 859 slivers of a BMW, seen almost edge-on by their own projection, produced Jacobian columns twenty orders of
magnitude too long and drew white spikes metres long.

**Width.** The original's $\sigma = 0.65$ leaves a converted surface about 30 % transparent to a ray tracer. Measured
on a unit quad lit head-on (mesh pixel 0.5032):

| $\sigma$ | coverage | pixel |
|---|---|---|
| 0.65 | 0.702 | 0.351 |
| 0.85 | 0.848 | 0.425 |
| 1.0 | 0.911 | 0.456 |
| 1.2 | 0.957 | 0.479 |
| 1.5 | 0.986 | 0.493 |

The default is therefore $\sigma = 1.0$; `--sigma 0.65` restores the original.

**Flatness and surfels.** The original wrote $10^{-7}$ model units as the third size, which made a Gaussian's
thinness depend on the model's scale: a 66 mm pawn traced correctly while a 100-unit fox became a ghost under the ray
tracer. The third size is now the fraction $\varphi$ (default 0.1) of the smaller in-plane size. `--flatness 0`
writes surfels — Gaussians with one scale exactly zero, the 2D Gaussian-splatting disc — which the rasteriser draws
by a ray–plane intersection per pixel and the path tracer clamps to $10^{-4}$ of the widest axis.

**Opacity as coverage.** A point of a converted surface lies under several Gaussians at once, so an opacity written
straight into each over-covers: a mask of 0.5 covered 96 % of what stood behind it. Every opacity the conversion
reads (`--opacity`, the material's constant, a map's value, what a glass keeps) is treated as the surface's
*coverage* $F$ and converted into the per-Gaussian $\alpha$ by expanding the transmittance of the stack to second
order: on the cell lattice the footprints sum to $2\pi\sigma^2$ and their squares to $\pi\sigma^2$, so

$$
\tfrac{\pi\sigma^2}{2}\,\alpha^2 + 2\pi\sigma^2\,\alpha = -\ln(1 - F)
\;\;\Longrightarrow\;\;
\alpha = \frac{\sqrt{4S^2 + 2S\,L} - 2S}{S} + \tfrac{1}{255}, \quad S = \pi\sigma^2,\; L = -\ln(1-F),
$$

the $1/255$ paying back the area a Gaussian loses where it is cut at $1/255$; $F \ge 0.999$ maps to 1 and 0 to 0.
Measured on planes of constant coverage 0.25 / 0.5 / 0.75, the raster gives 0.251 / 0.501 / 0.752 and the ray
tracer the same at full size, against 0.253 / 0.510 / 0.770 with the first-order term alone. A cut-out map below
`--opacity-cut` (0.5) writes no Gaussian at all, so the budget goes where the surface is.

**A triangle that catches no cell.** A triangle smaller than a cell need contain no cell centre; rasterised, it
vanishes. Converted, that removed the Buick's grille and window rubbers (80 137 triangles became 34 011 Gaussians).
Such a triangle now gets one Gaussian at its centroid, sized by its own second moments (`m2sOwnSize`): the uniform
triangle's covariance $\Sigma_\triangle = \tfrac{1}{12}\sum_k \mathbf d_k \mathbf d_k^\top$ (corners about the
centroid, in the tangent plane), widened as a cell is, $\Sigma = 12\sigma^2 \Sigma_\triangle$, with the wide axis
clamped to $[0.05\,s_{\max}, s_{\max}]$ of the cell's size and the narrow axis to at least 0.02 of the wide. Before
this, a panel's rolled lip — a strip of sub-cell triangles — drew a band of cell-wide beads three pixels thick along
every Corvette edge where the path tracer draws a hairline. In the Corvette bakes, glass and tyres come out at about
one Gaussian per triangle at both 640 and 1280 px.

**Deterministic order.** The original appended with an atomic counter, so two conversions of the pawn produced the
same 729 073 Gaussians in different orders. The emit is now three passes — count each triangle's covered cells, an
exclusive scan in one 256-thread workgroup, write each triangle's Gaussians at its offset — sharing one function
(`m2sTriangleOf`) so that counting and writing cannot disagree. Two conversions are now byte-identical, a budget
keeps the first Gaussians in mesh order, and Gaussian $i$ is the same Gaussian from one pose to the next. A record
that decodes to nothing is written as an empty slot rather than dropped, so array indices stay aligned.

## 3.5 Where a surface stops or folds

A Gaussian is a cell wide, and on the Corvette a cell is a pixel of the conversion camera. A part narrower than that
— the flange a hood folds down into its gap — still received cell-wide Gaussians in its own plane; they reached out
of the gap over the paint beside it, carrying the gap's baked darkness, and every panel edge became a thick black
wavy line where the path tracer draws a thin one.

**Hard edges** (`mesh2splat_edges.slang`). Per mesh, over all its pieces, every edge and corner is keyed by where its
points stand and radix-sorted on the device; each run of equal edge keys is one edge. An edge is *hard* when it is
open (one triangle uses it: the part stops or another mesh begins), when the two triangles sharing it fold by more
than $40^\circ$ (`kCreaseDegrees`), or when more than two triangles share it (non-manifold). Because keys are
positions over the whole mesh, a boundary between two materials on one smooth surface is not an edge. Each hard edge
becomes a line with an inside, $(\hat{\mathbf m}, c)$ with inside $\hat{\mathbf m}\cdot\mathbf x \ge c$ — the plane
through the edge, orthogonal to its triangle, facing the third corner — known to its own triangle as a bit and told
to every triangle touching either end (up to five lines each), since a Gaussian near a corner can stand beside the
next triangle's hard edge.

**Holding the reach** (`m2sHoldEdges`, `--edge-reach` $\rho$, default 1.5). For each line in the Gaussian's plane,
let $\mathbf m$ be its normal in the tangent basis, $C$ the in-plane covariance, $r^2 = \mathbf m^\top C\,\mathbf m$
the variance across the line and $d$ the centre's signed distance to it. The size allowed across the line is
$a = \max(d/\rho,\; 0.05\,r)$, and where $r > a$

$$
C \leftarrow C - \kappa\,(C\mathbf m)(C\mathbf m)^\top, \qquad \kappa = \frac{1 - a^2/r^2}{r^2},
$$

which makes $\mathbf m^\top C\,\mathbf m = a^2$ exactly while keeping the variance along the line's conjugate
direction, so a strip of Gaussians still covers the strip. Lines not lying in the Gaussian's plane (the flange seen
from the paint it folds off), and lines the centre stands more than a quarter size behind, are ignored; lines are
applied in a canonical order so the result does not depend on discovery order. The new covariance is
re-diagonalised into a tangent and two sizes. The count of Gaussians is unchanged — only their shapes are.

**Closed shells.** The same pass counts edges not shared by exactly two triangles, one counter read back per mesh. A
mesh is marked closed when

$$
n_{\text{open}} < 2\sqrt{1.5\,T},
$$

$T$ its triangles ($1.5T$ approximates its welded edges), and its Gaussians carry
`primvars:athenea:splat:closed = 1`, which allows a player or a cut to drop nodes facing away. A fixed share of edges
was tried first (2 %) and failed: an open sheet's border grows as the square root of its area, so a fine enough tube
passed as closed — the rims' barrels (1 %) were culled and light showed between the spokes on the phone. Against the
square root, the doors (no open edge), logos (1.4) and calipers (1.5) stay closed and rims (4.0), tyres (3.1), hood
(3.1) and bumper (4.4) do not: 102 of the Corvette's 192 meshes. A car's door is two skins; drawn without culling,
the inner skin sorted in front of the outer in patches, a net of dark cracks at phone size.

**Measured.** The Corvette's TX base at 640 px of `/World/Camera`, rebaked with edges held (default 1.5) and as a
control with `--edge-reach 0`, both 5 380 804 Gaussians; day frame at 1920 × 1080 against the mesh path trace:

| bake | relMSE (HDR) | PSNR (display) | SSIM |
|---|---|---|---|
| previous base (old build) | 0.0900 | 21.83 dB | 0.8555 |
| control, edges not held | 0.0902 | 21.86 dB | 0.8554 |
| edges held, $\rho = 1.5$ | 0.0887 | 21.93 dB | 0.8567 |

The global gain is small because the edges are a small fraction of the frame; the visible effect is on the panel
gaps.

<!-- /Users/muriel/luc/athenea-renders/simplify3/edges/edges_cmp.png -->
<!-- /Users/muriel/luc/athenea-renders/simplify3/edges/day_new_Camera.png -->
<!-- /Users/muriel/luc/athenea-renders/simplify3/edges/day_ctl_Camera.png -->

## 3.6 Material capture

Each Gaussian record carries position, coverage-derived opacity, three sizes, a rotation and a colour, followed —
depending on what the cloud needs — by the shading normal and metallic, roughness, transmission and texture
coordinates; four joints and their gradients; three relief entries for displacement; an emission entry; four layer
entries; and the shape operator.

**Base.** The colour is the material's base colour times its base weight, times the albedo map where there is one
(decoded to linear). Metallic and roughness are the material's constants, multiplied by the packed map's channels
where there is one (blue metallic, green roughness), as glTF multiplies its factors; a missing channel is a factor
of one. The original defaulted to $(0.1, 0.5)$ from the map alone, so every material without a map became the same
plastic; with the material's own values passed, a converted quad under a head-on light reads 0.5009 against the
mesh's 0.5032.

**Shading normal.** A normal map is read in the frame its UVs give — $\partial\mathbf p/\partial u$ from the
triangle's positions and UVs, orthonormalised against the normal, as the mesh shader does — and the result is kept as
the Gaussian's shading normal (`primvars:athenea:splat:normal`) beside its geometric frame. Previously the map was
read against the longest edge, and every facet of the pawn's marble turned its veins by its own angle. The disc
itself is not turned unless `--normal-map-turns` is given, which would open the surface where the relief is steep.

**Layers** (proposal 026). A Gaussian carries the specular weight, colour and index; the coat's weight, roughness and
index; the sheen's colour times its weight and its roughness; and the coat darkening switch — OpenPBR's units, three
words per splat on the device. Up to three maps on these inputs are sampled per Gaussian (`m2sLayersAt`). This was
needed because the Corvette's paint (`Car_Paint_Main`) is OpenPBR metalness 1 over a base of about 0.05 under a
clear coat of weight 1: everything the eye sees of it is the lacquer, and without the coat a converted car came out
nearly black.

**Schlick metals.** OpenPBR and glTF metals are MaterialX `generalized_schlick_bsdf` (colour head-on, specular colour
at $82^\circ$), not a conductor of the artistic index; for a dark metal the two differ by a factor of two at grazing
angles (at $60^\circ$: 0.172 for the conductor, 0.077 for the Schlick from $F_0 = 0.047$). The material's kind is
written per Gaussian (`primvars:athenea:splat:schlickMetal`) and the renderer reflects it with the DFG fit of its two
ends. An earlier variant of the same confusion made the bake drop every OpenPBR metal as dielectric polish: a
Mustang's front bumper stored the constant that decodes to black in all 27 016 Gaussians; with Schlick lobes of
$F_0 > 0.2$ kept as body, the bumper's mean colour became (0.45, 0.51, 0.61).

**Emission and transmission.** Emission is the material's colour times its map, linear and unbounded, in an entry of
its own. Transmission $T$ and transmission colour $\mathbf c_T$ are a channel the original did not have (glass came
out opaque white). The colour is taken towards the transmission colour,

$$
\mathbf c \leftarrow \operatorname{lerp}\!\bigl(\mathbf c,\; \operatorname{lerp}(\mathbf 1, \mathbf c, w_T)\,\mathbf c_T,\; T\bigr),
$$

with $w_T = 1$ for glTF (transmission tinted by the base colour) and 0 for OpenPBR and standard_surface. The opacity
is the coverage mapping of Section 3.4 applied to $o\cdot o_{\text{mat}}\cdot\text{mask}\cdot
\operatorname{lerp}(1, g, T)$, $g$ being what a fully transmitting surface still covers (Section 3.8).

**The material roughness mark.** The stored roughness may be widened by the specular filter (Section 3.7). The
material's own roughness before widening is written separately, `primvars:athenea:splat:materialRoughness` (−1 under
a roughness map, read as sharp). The player uses it to decide whether the polish lobe answers its occluded part from
the baked reflected field (material roughness at least 0.3: paint, 0.34) or keeps it dark (chrome 0, rough metal
0.22); in the decimated tree it is packed into the `pbr` word (bit 26 present, bits 27–31 the value in five bits).

## 3.7 The specular filter (Toksvig widening)

A Gaussian has one normal. Where a strip of chrome curves under it, the path-traced pixel averages a highlight that
moves across the footprint and shows a thin line; the Gaussian shows the light its single normal meets, at full
brightness, over its whole footprint. On the Corvette the grille trim — four pixels high — came back as a row of
white beads, and the chrome read 1.6 times the path trace while a chrome ball validated within 1 %.

The conversion therefore widens the roughness by the spread of normals under the Gaussian (`m2sSpread`,
`m2sWidened`). The turn rate is the largest angle between corner normals per unit corner distance, over the
triangle's three edges, and the spread is that rate times the Gaussian's wider size:

$$
s = k\,\max_{e}\frac{\angle(\mathbf n_{e_0}, \mathbf n_{e_1})}{\lVert \mathbf p_{e_1} - \mathbf p_{e_0} \rVert}\,\max(s_u, s_v),
$$

with $k$ = `--specular-filter` (1 with `--transfer`, 0 otherwise). With perceptual roughness $r$ and GGX
$\alpha = r^2$, the spread is added to the slope variance (Toksvig):

$$
\alpha'^2 = \alpha^2 + 2 s^2, \qquad r' = \operatorname{saturate}\!\left((r^4 + 2 s^2)^{1/4}\right).
$$

A flat triangle, or one whose corners share a normal, is unchanged. The coat's roughness is widened the same way
unless the cloud carries the shape operator.

**The shape operator.** For a transfer the conversion also writes, in the Gaussian's own axes $(\mathbf u, \mathbf
v)$, the symmetric shape operator $S$ with $\mathrm d\mathbf n = S\,\mathrm d\mathbf x$, solved from the corners'
normal differences against their position differences (`m2sShape`, stored as $(S_{uu}, S_{uv}, S_{vv})$). The
renderer uses it to turn the reflection across the footprint pixel by pixel (the per-pixel "sharp lobe"). Widening
the coat as well would blur twice, so where $S$ is written the coat is not widened, and the player reads the sharp
lobes at the unwidened roughness $r_0 = \max(r'^4 - 2s^2, 0)^{1/4}$ (`unwidened`), with $s$ carried per Gaussian.

**Measured.** At 640 px the coat came out at roughness 0.17–0.24 with 5–14 % high-pass on folds; at 1280 px the
spread under each Gaussian is smaller and the coat's roughness comes out at exactly 0, so the widening matters less as
the bake gets finer. Turning it off is nevertheless worse (Section 3.11): with `--specular-filter 0` a dashed dark
line appears on the sill's fold and the chrome strip sparkles.

## 3.8 Glass

A Gaussian cannot refract, but it can let what is behind it through by its opacity, reflect through its specular
lobe, and carry an index of refraction that the renderer bends by (`primvars:athenea:splat:ior`, one per cloud; with
layers each Gaussian bends by its own specular index). The conversion's job is to decide, per transmitting piece,
whether the glass is a *solid* (a body: a ball, a lens) or a *thin wall* (a sheet or slab whose two faces bend
nothing), because the two are drawn differently.

**Classification** (`classifySheets`, `mesh2splat_sheet.slang`), for every transmitting piece whose material does
not already declare itself thin-walled. On the device, the piece's edges are keyed by position, sorted, and their
runs counted; the signed volume $V$ (tetrahedra to the origin) and the area $A$ are summed. Then

- *open*: more than $1/200$ of the edges have one triangle, or any edge has more than two;
- *slab*: closed, and the thickness $2V/A$ is below four of the model's cells or below $0.02\sqrt A$
  ($2V/A$ is a slab's gap and two thirds of a ball's radius).

An open or slab piece is read thin-walled — unless it is *tinted* (transmission colour luminance below 0.9), which
stays solid for now, since a thin wall lets the background through grey. The Corvette's glass turned out to be
closed slabs 3–4 mm thick (research proposal 055), which the edge rule alone called solid; the thickness rule reads
them. Each decision is printed, e.g. for the rear light lens: "tinted, kept solid glass: 200 of its 17284 edges open,
0 shared by more than two; 1.199 thick (2V/A) against a cell of 0.007921". `--thin-glass MAT` and `--solid-glass MAT`
override per material.

**Coverage.** A solid glass covers `--glass-opacity` of what is behind it (default 0.6; a TX lens answers its
background from the domes alone, so a glass that covered fully hid every other light shining through it). A thin
wall covers exactly what the sheet reflects head-on, both faces with inter-reflection:

$$
g_{\text{thin}} = \min\!\left(\frac{2R}{1+R},\; 0.998\right), \qquad R = \left(\frac{n-1}{n+1}\right)^2,
$$

and the frame scales its reflection back by $1/R$. At the windscreen's index of 1.16 that is an opacity of 0.0055,
and the rasteriser draws such sheets at an opacity of at least 0.1 with colour scaled down by the same factor, which
keeps the product $\alpha\,\mathbf c$ — the reflection — intact (windshield relMSE 2.26 → 0.89 under the workshop
sky).

## 3.9 Shadow and light catchers

The ground is not converted as a surface; its interaction with the object is converted instead, so the playback is
Gaussians only.

**Shadow catcher** (`--shadow-catcher --prim OBJECT`). The ground is found automatically (the largest flat mesh
outside the object whose top is at the object's bottom; `--catcher-ground` names one). A patch of quads — the
object's footprint widened by `--catcher-margin` heights (1.5), split into a grid so that no triangle exceeds
`--max-cells` — is added to a session copy of the stage and converted alone at a cell of a hundredth of the object's
height, as a TX transfer whose rays meet the object and the ground. At draw time the gaussian's opacity is
$-\ln(\text{ratio})/2\pi$, where the ratio is light reaching it with the object against without, since a lattice of
$\sigma = 1$-cell Gaussians sums to $2\pi$ and a layer of opacity $a$ leaves $e^{-2\pi a}$. The Corvette's: a
5.9 × 8.3 m patch, 11 × 11 quads, 309 k Gaussians at a 1.26 cm cell, baked in 79 s. Against the path trace:

| | under the car (GT 0.182) | bumper contact (GT 0.043) | open ground (GT 0.217) |
|---|---|---|---|
| mesh ground's own shadow map | 0.182 | 0.041 | 0.215 |
| catcher, depth 2.4, two triangles | 0.144 | 0.002 | 0.206 |
| catcher, depth $2\pi$, grid | 0.192 | 0.030 | 0.220 |

**Light catcher** (`--light-catcher`). The same patch, reaching `--light-reach` object lengths (3) ahead and behind,
at a cell of a thirtieth of the height, converted as a radiance bake per light group with the dome black; drawn
additively, it carries the low beam's pool and the tail lamps' glow.

## 3.10 Validation against the path trace

`--validate DIR` measures a conversion material by material against the stage path-traced, for any stage. The groups
are the stage's material bindings (a mesh of two materials is in both); the conversion is the command's own, run
once per group with every mesh outside it hidden, so every option is what is measured. The ground truth is path
traced once (`--validate-paths` 512, `--validate-bounces` 6) and kept; the stage is rasterised once as meshes with
its Cryptomatte, and each group's mask is the coverage of every rank whose id hashes one of its meshes. Within the
mask's box the Measure effect gives relMSE, p99 and the means of the cloud and of the mesh raster beside it, written
to `validate.json`. `--validate-sky white` (or an image) draws everything under another sky with the stage's other
lights off, so a transfer's error is its occlusion's and reflection's alone. It is what found, among others, that a
solid glass ball traced at half the path-traced brightness (0.096 against 0.195) before rays were bent into it, and
that the paint ball under a white dome read 13 % under because of the coat's Fresnel.

## 3.11 Measured results on the Corvette

All bakes below are TX transfers, `--cell-from-camera /World/Camera`, 64 paths per Gaussian, 3 bounces, no adaptive
extra paths, with the specular filter and edge reach at their defaults (1 and 1.5). "Car crop" is display PSNR on the
900 × 420 crop of the car, "frame" on the full 1920 × 1080 frame, both against the mesh path trace, after the merge
tree is cut at 0.25 px with back culling.

**640 → 1280 px.** The 640 px bake's leaves were 3–4 mm on the hood, and the door's stripes and the sill's hairs —
absent from the path trace — were in the cloud. The car was re-baked at 1280 px in 8 groups of top-level parts
balanced by count, each converted with all other parts hidden (part names taken from the Cryptomatte manifest). The
eight bakes wrote 891 387, 896 496, 809 073, 790 403, 1 631 765, 1 073 954, 1 302 878 and 1 307 473 Gaussians
(8 703 429 in total) in 212–304 s of transfer each, about 34 min altogether.

| | 640 px | 1280 px |
|---|---|---|
| leaves | 5.38 M | 8.70 M |
| hood / door | 50 k / 79 k | 151 k / 266 k |
| glass, tyres | about one per triangle | about one per triangle |
| car crop / frame (PSNR) | 16.55 / 22.18 dB | 17.88 / 23.90 dB |

Analysis of the 1280 px cloud against the 640 px one: hood highlights 3.2 → 2.0 px off their path-traced position,
the door's open-reflection contrast 0.35/0.53 → 0.56/0.74 of the path trace's, the grille's holes in the coarse cut
2.24 → 1.04, the sill's hairs 987 → 459 px from the resolution alone.

**The A/B/C test.** Before committing the 1280 px settings, eight parts (both doors, both sill panels, front fender,
chrome grille, grille rim, hood) were baked with the whole car casting, 931 017 Gaussians in each variant:

| variant | settings | result |
|---|---|---|
| **A** | as the 640 px bake ($k = 1$, $\rho = 1.5$) | chosen: no hairs, the door smooth |
| B | `--specular-filter 0` | dashed dark line on the sill's fold; the chrome strip sparkles |
| C | B + `--edge-reach 1.0` | elongation and hairs identical to B |

The hypothesis under test was that the door's streaks were the coat's Toksvig widening saturating on folds and the
sill's hairs the edge pass's elongated Gaussians. Neither held at 1280 px: the widening is needed on folds and the
tighter edge reach changed nothing, so A was kept.

<!-- /Users/muriel/luc/athenea-renders/pieces/car1280/test/A.png -->
<!-- /Users/muriel/luc/athenea-renders/pieces/car1280/test/B.png -->
<!-- /Users/muriel/luc/athenea-renders/pieces/car1280/test/C.png -->

**The 5120 px close-up level.** Close up on the phone the camera stands far nearer than `/World/Camera`, about
0.5 mm per phone pixel against 3–4 mm leaves. A logo-and-grille test from a close camera grew from 108 527 to
404 594 Gaussians between 1280 and 5120 px and showed the checkered flag, the bowtie and the grille's mesh crisp. The
66 parts the 1280 px bake had already given at most 1.5 Gaussians per triangle (badges, grilles, lamps, calipers,
mirrors, wheels, wipers, trims) were re-baked at 5120 px in seven groups; the panels at 5120 px or as fine as kept a
part under a cap of leaves (5 M, lowered to 4.5 M after a 5.01 M tree did not fit in device memory), with
`--max-cells 1048576`:

| groups | leaves | resolution (px) |
|---|---|---|
| fine0–6 (66 detailed parts) | 18 610 627 | 5120 |
| door (pilot) | 4 084 170 | 5120 |
| Bottom | 4 897 428 | 3392 |
| Part_3 | 4 370 459 | 4480 |
| Infinite_Black | 4 481 738 | 4736 |
| six large panels and the windshield | 15 994 566 | 5120 |
| pan0–4 (small panels) | 17 088 152 | 5120 |
| Front_Bumper_001 | 5 008 986 | 4480 |
| **24 trees** | **74 536 126** | |

Each large panel's transfer took 2–25 min (e.g. the hood, 2 264 198 Gaussians in 120 s; Front_Bumper_001,
5 009 059 in 1 487 s) on a machine shared with other GPU jobs, so these are not benchmarks. All 24 merge trees were
written (74 328 031 merges, 208 095 roots, 148 864 157 nodes in 49 634 pages, 87.4 GB) and share one occlusion
codebook of 8 182 entries; Section 6.9 gives the details. (The baked count of Front_Bumper_001, 5 009 059, is the
conversion log's; the tree holds 5 008 986 leaves, the figure used in the table.)

## 3.12 Limitations

- A Gaussian has one normal and one cell's footprint: a chrome tube a few pixels across is still 38 % brighter than
  the path trace on the front bumper, and the bumper's lower lip is ragged.
- Lamps carry no emission in the transfer bake; headlights are dark against the path trace.
- A tinted thin glass is converted solid, and a cloud keeps a single index unless it carries layers.
- The coverage mapping assumes the cell lattice's overlap $2\pi\sigma^2$; sub-cell triangles and displaced, split
  cells overlap otherwise and are not corrected.
- Nested point instancers are expanded only at the innermost level; motion authored as velocities is read at the
  sample.
- Device memory bounds a part at about 5 M leaves for the merge tree that follows; larger parts are converted at a
  coarser camera resolution.
