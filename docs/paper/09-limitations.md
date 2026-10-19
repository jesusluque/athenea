<!-- Copyright (c) 2026 jesus luque. -->

# 9. Limitations and future work

The limitations below are gathered from the sections' own lists (§2.16, §3.12, §4.12, §5.13, §6.11, §7.13) and
grouped by where they act, the relit bake first and the mobile application last. Each is stated with the figure that
measures it, where one exists.

::: {.nota}
**How the list is organised** — Each limitation is classified by the stage where it arises: the bake (what the cloud cannot store), the representation (what a Gaussian cannot express), animated assets, the engine, levels of detail and the player, and finally what the evaluation has not measured. Distinguishing *what limits the representation* from *what limits the current implementation* matters when reading each row: a representation limitation (one normal per Gaussian, a rank-one field) is lifted only by changing what is stored and paying in memory; an implementation one (the TLAS rebuilt every frame, synchronous compilation) is lifted by engineering work without changing the format.

*References:* §2.16, §3.12, §4.12, §5.13, §6.11, §7.13 of the paper; `docs/decisions.md` (the "Not done" and "Not done, not verified" subsections of each section).
:::

## 9.1 Remaining defects on the Corvette

These are what the relit 1280 px car still shows against the path trace.

| defect | measured | cause | remedy proposed | § |
|---|---|---|---|---|
| lamps dark | headlights dark against the path trace | the transfer bake carries no lamp emission | bake the lamps' emission and their light on the car | 3.12, 4.12 |
| chrome strip on the front bumper | 38 % brighter than the path trace, with a lower peak | a tube a few pixels across: each Gaussian reads its own mirror with one normal | — (open) | 3.12, 4.12 |
| rim's near-ground reflection | 0.72 of the path trace | — (not analysed further) | — (open) | 4.12 |
| door's open reflections | 1.13 of the path trace; 68 % of the excess where the mirror meets the car | 16 × 16 cells of about 11° let sky through behind a mirror or pillar | 32 × 32 cells (+96 B a Gaussian) would roughly halve it | 4.10.4, 4.12 |
| hood highlights | 2.0 px off their path-traced position | the per-pixel turn extrapolates the centre triangle's shape operator | a curvature fitted over the footprint from mesh samples ($p_{90}$ 9.8° against 46°) would roughly halve it | 4.10.6, 4.12 |
| bumper's lower lip | ragged | one cell's footprint per Gaussian at a rolled edge | — (open) | 3.12 |
| grille's holes in coarse cuts | 2.24 of the leaves at 640 px, 1.04 at 1280 px | merges of sub-pixel strips; the hole is a property of the projection | finer leaves (done at 1280 px); no view-free per-node measure sees it | 6.5, 6.11 |
| far, thin trims | fat dashes in every cut | the error bounds miss this error | — (open) | 6.11 |

::: {.nota}
**Dark lamps** — The transfer stores how each Gaussian responds to *incoming* light (the sky, the sun and the stage's lights through the open directions); a lamp's emission is a source term $L_e$ the bake does not carry, and neither is its light on the car (the reflector, the lens, the nearby hood). Since a car's lamps are fixed relative to the car, their contribution is static in the object's frame and could be baked as a constant term per Gaussian, switchable by light group (the light-group mechanism already separates the dipped beam and the tail lights for the ground's light catcher, §3.9).

**Bumper chrome strip** — A chrome tube a few pixels wide: across the tube the normal turns by up to 180°, and the reflected direction turns twice as much. A Gaussian that covers that width with a single normal reflects a single point of the environment instead of the arc the tube sweeps; Toksvig widening (§3.7) turns that arc into a higher roughness, which gives the correct mean but flattens the peak (38 % brighter and with a lower peak). It would take a normal that varies non-linearly within the Gaussian (a cylindrical model of the tube) or Gaussians narrower than the tube.

**Door's open reflections, 16 × 16 cells** — The open directions are an octahedral grid over the sphere: 16 × 16 = 256 cells of about $4\pi/256\approx0.049$ sr each, of the order of 11–13° on a side. A mirror or pillar narrower than a cell leaves the cell "open" and the reflection shows sky where the path trace shows car (68 % of the excess of 1.13). 32 × 32 is 1024 cells of about 5.6°: 1024 bits instead of 256, that is 32 integers instead of 8, the +96 B per Gaussian of the table.

**Hood highlights** — The per-pixel turn extrapolates the normal across the Gaussian with the shape operator $S$ (the derivative of the normal with respect to position on the surface) of the centre triangle: $n(x)\approx n_0+S\,x$. Outside that triangle the curvature changes, and 64 % of the hood's footprints cross an edge. Fitting $S$ by least squares to mesh samples over the whole footprint lowers the $p_{90}$ error from 46° to 9.8° (table of §4.10.6).

**Lower lip, grille holes, thin trims** — An edge rolled over on itself has two surfaces within one cell. The grille's holes depend on the projection: a merge of two sub-pixel strips fills the gap between them, and no per-node measure that does not depend on the view can know when that gap is visible. The far, thin trims have the same problem: the cut's error bounds measure how far mass moves, not whether a thin line widens.

*References:* [20], [15], [35] in §10; Cigolle et al., "A Survey of Efficient Representations for Independent Unit Vectors", JCGT 3(2), 2014 (the octahedral map); M. do Carmo, *Differential Geometry of Curves and Surfaces* (1976), for the shape operator. Code: `shaders/athenea/splat/splat_relight.slang`, `shaders/athenea/splat/splat_project.slang`; `docs/decisions.md` (sections "Step 1: a reflection's occlusion has a direction", "A reflection off a curved strip is the mean over the gaussian (Toksvig)", "A TX transfer's reflection turns across the gaussian (proposals 044, 046, 049)", "A car's lamps switched at run time: light groups (proposals 062, 066)").
:::

## 9.2 The representation and the transfer

- **Rank-one field.** The reflected field's coupling to the sky puts a lit ground's colour, not its shape, into a
  reflection under a sky lit from one side; at degree 3 it resolves a horizon to about twenty degrees (correlation
  0.56 at degree 3, 0.63 at degree 5, 0.85 with a 32 × 32 octahedral map, in simulation) (§4.12).
- **Lights inside the scene.** One sun is extracted per dome. Lamps inside a scene are shadowed by bits that say which
  directions leave the scene, not which reach the lamp (§4.12).
- **One normal, one footprint.** A Gaussian has one normal and a cell's footprint; thin curved strips and rolled
  edges are where this shows (§9.1). The coverage mapping assumes the cell lattice's overlap $2\pi\sigma^2$, which
  sub-cell triangles and edge-held Gaussians do not have (§3.12).
- **Glass.** A tinted thin glass is converted solid, and a cloud keeps a single index of refraction unless it carries
  layers (§3.12). Thin-sheet Gaussians are lifted to a 0.1 opacity floor per Gaussian, so merging fewer, larger
  sheets varies the transmission cell to cell; a floor per pixel is pending (§6.11).
- **Size on device.** Device memory bounds a part at about 5 M leaves for the merge tree; the 4.5 M cap made larger
  parts convert at a coarser camera resolution (Bottom at 3392 px, Part_3 and Front_Bumper_001 at 4480 px; §3.11,
  §6.9).
- **Instancing.** Nested point instancers are expanded at the innermost level only, and motion authored as velocities
  is read at the sample (§3.12).

::: {.nota}
**Rank one** — The reflected field stores, for each Gaussian's closed directions, what those directions show, *coupled* to the sky: one value per Gaussian multiplied by a function of the current sky. As a function of two variables (position and sky) it is a separable product $f(x)\,g(\text{sky})$, that is, of rank one: when the sky changes, the reflection changes in brightness and colour, but the *shape* of what is reflected (where the ground is lit) does not. Under a sky lit from one side, the real ground has a lit area and a shadowed one; the rank-one field puts in its mean colour.

**Degree of the harmonics** — Spherical harmonics up to degree $\ell$ form a band-limited basis: the smallest angular structure they represent is of the order of $\pi/\ell$, and a step (a horizon) reconstructed with them widens and rings (the Gibbs phenomenon). The simulated correlations (0.56 at degree 3, 0.63 at degree 5, 0.85 with a 32 × 32 octahedral map) show that raising the degree yields little and that an explicit map yields a lot, at a much higher memory cost. The middle way would be a higher rank: several shared basis functions (as in CPCA [15]) instead of one.

**Lights inside the scene** — The open directions say which directions *leave the scene*, not whether a direction reaches a lamp located inside it. For a sun or a distant lamp this is the same thing; for a lamp surrounded by objects, an object beyond the lamp in the same direction wrongly "shadows" it. In addition, a single sun is extracted from each dome (the brightest texels form a disc that is lit as an analytic light). Storing depth per cell (at what distance the direction closes) or baking the visibility of each fixed lamp would lift it.

**One normal, one footprint, and coverage** — The coverage of a lattice of Gaussians: an unnormalised Gaussian $g(r)=e^{-r^2/2\sigma^2}$ integrates to $2\pi\sigma^2$ over the plane, and on a lattice of pitch 1 cell with $\sigma=1$ cell the sum at each point is approximately $2\pi$. With opacity $a$ per Gaussian, the transmittance is $\prod_i(1-a\,g_i)\approx e^{-a\sum_i g_i}=e^{-2\pi a}$, and from this follows the opacity that reproduces the material's coverage. A triangle smaller than a cell, or a Gaussian held at an edge, has no neighbours to complete that sum, and its coverage falls short.

**Glass** — A tinted thin glass converted as a solid refracts like a thick lens; a cloud with a single index of refraction cannot mix materials of different index except by layers. The 0.1 opacity floor per thin-sheet Gaussian adds up non-linearly: two sheets of transmission $0.9$ give $0.81$, one merged sheet lifted to 0.1 gives $0.9$, and the transmission varies from cell to cell depending on how many sheets there are. A floor applied per pixel, after compositing, would fix it.

**Size on device and instancing** — A part's merge tree is built entirely on the GPU: at ~5 M leaves the working forest does not fit, and the larger parts were converted at a lower resolution. Building in pieces and joining the trees would lift the cap. In instancing, only the innermost point instancer is expanded, and motion authored as velocities is read at the sample's instant instead of extrapolating $p+v\,\Delta t$.

*References:* [14], [15], [16], [17], [43] in §10; Ramamoorthi and Hanrahan [17] on the band limit of lighting in SH; code: `shaders/athenea/technique/env_sun.slang`, `shaders/athenea/usd/transfer_field_fill.slang`, `shaders/athenea/usd/mesh2splat_sheet.slang`; `docs/decisions.md` (sections "Step 2: what the closed directions show", "Step 3: degree 3, and a maximum level to measure against", "Step 4: lights that are not the sky", "The sun comes out of the sky and is lit as a light", "A thin sheet of glass is drawn as a layer", "A windscreen modelled as one surface is a sheet (`--thin-glass`)", "A converted surface covers what its opacity says").
:::

## 9.3 Animated assets

- Four influences per Gaussian truncate the sparrow's wing tips (27 % of wing vertices carry more); the share of
  truncated Gaussians is unmeasured (§5.3, §5.13).
- No dual-quaternion skinning for clouds, no retargeting and no clip blending; several skinned meshes at different
  prim transforms keep only the first one's (§5.13).
- The traced route poses once per frame, so it does not blur a skinned cloud, and LoD instances carry no motion
  (§5.13).
- Per-part visibility leaves out a part's own self-shadowing, assumes rigid parts that occlude independently, and has
  not been baked beyond about 3 M Gaussians per part; zonal transfers keep only the direct half and the bind pose's
  self-shadowing (§4.12, §5.8, §5.13).
- The zonal fit's error on the sparrow, and a geometric measure of coverage at a close bend, are unmeasured (§5.13).

::: {.nota}
**Four influences** — In *linear blend skinning* (LBS) each point is transformed by $\sum_j w_j M_j$, with $M_j$ the skinning matrix of joint $j$. A Gaussian stores the four highest-weight influences, renormalised; a feather vertex of the sparrow has between ten and twenty-five, and truncating displaces the Gaussian relative to the mesh at the tips. The remedy is to store more influences (8, or a variable number with a start index), at more memory per Gaussian.

**Dual quaternions, retargeting, clip blending** — LBS collapses volume under large rotations (the *candy wrapper* effect); *dual quaternion skinning* (Kavan et al., 2008) interpolates rigid rotations and avoids it. *Retargeting* is applying an animation to another skeleton; *clip blending*, interpolating two animations. None of the three is present; all three are runtime work with no change of format.

**Motion blur on the traced route** — The rasteriser turns motion during the shutter into a rank-one covariance term (§2.5.3); the traced route poses the cloud once per frame (BVH refit), so for it the cloud is still during the shutter. It would need to pose at several instants or store the per-Gaussian displacement in the ray structure.

**Per-part visibility** — Each rigid part carries a visibility field that measures how it occludes the rest; the occlusion of several parts is composed as a product, $V\approx\prod_k V_k$, which is exact only if the parts occlude disjoint directions (if two parts cover the same direction, the product darkens it twice). A part is not shadowed by its own field, and zonal transfers keep only the direct half and the bind pose's self-shadowing.

*References:* [16], [41], [42], [43] in §10 (UsdSkel); L. Kavan, S. Collins, J. Žára, C. O'Sullivan, "Geometric Skinning with Approximate Dual Quaternion Blending", ACM TOG 27(4), 2008. Code: `modules/geom/src/Skinner.cpp`, `modules/usd/include/athenea/usd/StageRenderer.h` (`holdBindPose`, which documents the four influences), `modules/technique/src/SplatVisibility.cpp`, `shaders/athenea/usd/transfer_zonal_io.slang`; `docs/decisions.md` (sections "A gaussian carries the joints its triangle carries", "A cloud shadows itself by part: baked once, read every frame, no ray", "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame", "A posed cloud refits its ray tracing structure").
:::

## 9.4 The engine

- Splats are not yet inside the path tracer's secondary rays: with meshes, splats are composited over path-traced
  surfaces by depth and cast shadows onto them, but a bounce ray does not see them (§2.16).
- The rasteriser's motion and lens blur are moment-matched Gaussians, up to 38 % too peaked for long streaks (§2.16).
- Gaussians a few hundredths of a pixel across vanish from the raster; levels of detail are the remedy (§2.16).
- Raster mesh visibility pays about 8 ms of host time per 1800 draws; material compilation is synchronous; the TLAS is
  rebuilt every frame; lens distortion beyond $k_1, k_2$ is not applied (§2.16).
- The traced route reads a rotated cloud's transfer against a sky left in the world's frame (§4.12).
- **Hosts.** In Nuke an OpenFX node gives one layer a node, since Nuke takes only its own planes from one; the
  Cryptomatte manifest stops at the plugin, since OpenFX carries no metadata, so a matte is picked by id; the camera is
  linked knobs rather than the host's own camera; gaussians carry no normals or motion vectors as layers. The Blender
  add-on is built on an earlier engine and runs on macOS only (§2.14).

::: {.nota}
**Splats and secondary rays** — The mesh path tracer and the Gaussian ray tracer are distinct structures: a BVH of triangles and another of particles. With meshes in the scene, the engine composites the splats by depth over the path-traced image and uses them for shadows, but a bounce ray leaving a mesh queries only the triangle BVH. Integrating them requires the integrator, at each vertex, to query the particles as well and accumulate their transmittance along the segment, as 3DGRT [7] does for particle-only scenes.

*References:* [7] in §10; code: `modules/technique/src/PathTracer.cpp`, `modules/render/src/GaussianRayTracer.cpp`; `docs/decisions.md` (section "Splats shadow meshes in the path tracer").
:::

::: {.nota}
**Why 38 %** — A Gaussian of deviation $\sigma$ dragged at constant velocity over a length $L$ is its convolution with a box of width $L$. The rasteriser replaces it with a Gaussian of the same variance, $\sigma^2+L^2/12$ (the variance of a box of width $L$ is $L^2/12$). For long streaks ($L\gg\sigma$), the true profile is a plateau of height $1/L$ and the matched Gaussian has peak $1/\sqrt{2\pi L^2/12}$. The ratio is
$$\frac{1/\sqrt{2\pi L^2/12}}{1/L}=\sqrt{\frac{12}{2\pi}}=\sqrt{\frac{6}{\pi}}\approx1.382 ,$$
38 % more peaked: the limitation's figure follows exactly from matching moments. The same holds for bokeh (a disc). The remedy is to draw several Gaussians along the streak (more cost) or an analytic plateau kernel.

**Gaussians a few hundredths of a pixel across** — A projected Gaussian with a radius much smaller than a pixel falls below the rasteriser's opacity or size threshold and contributes nothing, even though the sum of many of them should cover the pixel. Mip-Splatting [4] handles it with a minimum-size 2D filter; here the remedy is the level of detail, which replaces thousands of tiny Gaussians with a pixel-sized merge.

**Host cost, compilation, TLAS and lens** — Raster visibility issues one draw per mesh and pays about 4.4 µs of CPU per draw; the standard remedy is indirect drawing with the draw list generated on the GPU. Compiling a material on the render thread blocks the frame in which it appears. Rebuilding the TLAS every frame costs in proportion to the number of instances; a *refit* when only transforms change is cheaper. Lens distortion is the radial model $p'=p\,(1+k_1r^2+k_2r^4)$; the Brown–Conrady model adds $k_3$ and tangential terms $p_1,p_2$.

**Rotated cloud on the traced route** — The rasteriser turns the sky lookup into the object's frame; the traced route reads the transfer in the object's frame against a sky fixed in the world, so a rotated object sees the sky rotated the other way. It suffices to rotate the lookup direction (or the SH coefficients, with Wigner rotation matrices) by the inverse of the object's transform.

**Hosts** — OpenFX is an image-effects API: a node returns the images the host accepts, without arbitrary metadata, so the Cryptomatte manifest (the hash-to-name table) does not travel and Nuke takes only the node's own layers. The host's camera is not exposed to an effect, which is why it is linked through knobs.

*References:* [4], [47] in §10; D. C. Brown, "Decentering Distortion of Lenses", Photogrammetric Engineering 32(3), 1966; J. Ivanic, K. Ruedenberg, "Rotation Matrices for Real Spherical Harmonics", J. Phys. Chem. 100(15), 1996; the OpenFX specification (openeffects.org). Code: `modules/render/include/athenea/render/Camera.h` (`distortionK1`, `distortionK2`), `modules/world/src/RayTracingScene.cpp`, `shaders/athenea/splat/sky_turn.slang`, `integrations/ofx`; `docs/decisions.md` (sections "A splat under a shutter: one rank-one term, not five passes", "Depth of field in the rasteriser: the other convolution", "The camera's lens", "The OpenFX plugin: athenea's gaussians in Nuke (18 Oct)", "Layers and the sky (19 Oct)", "A pixel says which prims it saw: Cryptomatte").
:::

## 9.5 Levels of detail, streaming and the player

- **Occlusion weighting.** The shared codebook is apportioned by entries used, not by how often a view culls a node;
  weighting by an orbit of views in the bake should lift culling past 84 % of the ideal (§6.11).
- **Not ported.** The closed-shell back rule and the error cut live in a development branch; the main engine's
  `CutSelector` is unchanged (§6.11).
- **Residency.** Pages hold about 4× the cut because needed and view-culled nodes share every page; the player splits
  traversal from payload rather than shrinking pages (§6.8, §6.11).
- **Merge opacity in the player.** The player draws each merge's mass, while athenea's rasteriser keeps the
  least-squares coverage; the coverage fit is the better choice except on the grille (§6.4).
- **The cut in the player.** The compact traversal does not keep a merge's LoD size, so a combined size/error cut has
  nothing to guard with, and the error cut alone did not beat the size cut in every view (§7.13).

::: {.nota}
**Weighting the codebook** — The occlusion codebook has 8192 shared entries; today every entry is worth the same. If they were apportioned according to how often a real view discards the nodes that use them (sampling an orbit of cameras in the bake), the entries would go where they save the most nodes. It is a weighted vector quantisation problem: k-means with frequency weights [23].

*References:* [23], [35] in §10; code: `shaders/athenea/lod/occlusion_mask.slang`.
:::

::: {.nota}
**Not ported** — The main engine's `CutSelector` (`modules/lod/include/athenea/lod/Lod.h`) selects the cut with the original rule; the error cut and the closed-shell rule are in the merge trees' development branch. It is an integration limitation, not one of method.

**Residency** — Pages group nodes by error band, not by visibility, so a needed page also brings in nodes the view discards. Shrinking the pages would improve the ratio at the cost of more requests; the player separates the traversal (small records, all resident) from the payload, so that the inefficiency affects only the cheap part.

**Merge opacity** — A moment merge conserves mass (the sum of opacities by area); the least-squares fit of §6.4 conserves coverage, the light the group actually blocks, which is what is seen. Mass counts the leaves' overlapping tails twice (a 27 % excess at the median, §6.4), and the fit $a^\star=\sum_x G(x)C(x)/\sum_x G(x)^2$ sets the merge's opacity to what its leaves really cover. The grille is the measured exception (41.16 dB with mass against 39.71 with coverage, close up at 393 px): there neighbouring merges overlap and composite as $1-\prod$, so any excess opacity darkens the whole patch. The player, a fork of Spark, draws the mass ($\min(\text{coverageMass},0.99)$); bringing coverage to the web format means changing one value per node when writing the tree (`--tree-coverage`).

**The cut in the player** — The compact traversal record (5–11 B) does not keep a merge's LoD size, so the player cannot combine "projected error" with "size on screen" to guard against large merges with low error; and the error cut alone did not beat the size cut in every view. Adding the size costs bytes per node in the traversal record.

*References:* [2], [11], [34], [35] in §10; code: `modules/lod/include/athenea/lod/Lod.h`, `shaders/athenea/lod/lod_cut.slang`, `shaders/athenea/lod/lod_pages.slang`; `docs/decisions.md` (section "Decimation: the levels of detail, cut once by what a merge loses").
:::

## 9.6 Gaps in the validation

- **One production asset.** The car-level figures are for one asset, the Corvette, with the sparrow and the pawn as
  the animated and rigid controls. Generality to other assets has not been measured.
- **One environment for the car.** The car-level comparisons against the path trace are under one sky
  (`autoshop_01_4k`, §1.6). Relighting fidelity under other skies and local lights has not been measured at the scale
  of the car.
- **The 5120 px close-up level against the path trace** is measured over the whole car at one camera and cut (0.25 px:
  24.00 dB on the frame, 18.31 dB on the car, §6.9) and on the close logo in the player (§7.12.3), not along a close
  orbit.
- **v8.2b on the phone.** The final data set is published and measured against the path trace in the player
  (§7.12.3); its download and frame times on the real iPhone are not, nor the 512-cluster basis there (§7.2.2).
- **Real-device timing** exists only for the stress ramps and two device tests (§7.13); the emulated phone's speed is
  not the phone's.

::: {.nota}
**External validity** — The car's figures are from one asset and one camera. The method was tuned by looking at this asset: each step of §4 was accepted because it improved the Corvette against its path trace. In statistical terms, the Corvette is both the tuning set and the test set (*in-sample*); the sparrow and the pawn are controls, not an independent (*held-out*) test set of production assets. Whether the changes generalise would be known by converting assets that played no part in the decisions, with the same metric.
:::

::: {.nota}
**One sky only** — The transfer's purpose is to relight: the cloud must be correct under skies and lights it has not seen. Measuring only under `autoshop_01_4k` checks that the bake reproduces itself under its own lighting, not relighting. The pawn was measured under two skies and with the sky rotated (§4.3, §8.4), and the harness has `--validate-sky` to draw everything under another sky (§3.10); it remains to do so at the scale of the car. The rank-one and interior-light limitations (§9.2) are exactly the ones that only appear under other light.

*References:* §3.10 and §4.3 of the paper; `docs/decisions.md` (sections "Measured: the pawn under two skies (M5 Pro, release)", "Under another sky: what was wrong, and the gate").
:::

::: {.nota}
**One camera against an orbit** — The cell size was derived from `/World/Camera` (`--cell-from-camera`), so that view is the one the bake favours: measuring only from it is measuring *in-sample* with respect to resolution. The close-up level exists precisely for close views, where a 1280 px leaf covers several pixels; an orbit of close cameras, with its own path trace per view, would measure whether the level holds there. 24.00 dB on the frame and 18.31 on the car against 23.90 / 17.88 for the 1280 px bake say that the level does not worsen the far view; they do not say how much it improves the close one.

*References:* §3.11 and §6.9 of the paper.
:::

::: {.nota}
**Real device** — The usable phone timing figures are those of the stress page and the two tests on the iPhone (§7.12.1); those of the emulated iPhone measure the player's logic with the desktop GPU (note of §8.1). v8.2b, the final 12.77 GB data set, has not been downloaded or timed on the phone; the download time depends on how many pages the cut requests, which is known only by measuring.

*References:* §7.12–§7.13 of the paper.
:::

## 9.7 Future work

In the order the measurements suggest:

1. Bake the lamps' emission into the transfer, so that the headlights and their light on the car match the path
   trace.
2. Fit curvature over each Gaussian's footprint from mesh samples, to halve the hood highlights' offset and steady the
   per-pixel turn.
3. Move to 32 × 32 open-direction cells where a sharp coat mirrors the car itself (the door's open class).
4. Address thin chrome tubes and rolled lips (the bumper's strip and lower lip), the rim's near-ground reflection and
   far thin trims.
5. Measure the relit car against the path trace under varied environments and local lights, and the 5120 px
   close-up level along a close orbit of the whole car.
6. For animated assets: more than four influences, cross-part occlusion in poses other than the bind pose, and
   visibility bakes beyond 3 M Gaussians per part.
7. Weight the occlusion codebook by an orbit of views, and port the error cut and the closed-shell rule into the main
   engine's `CutSelector`.
8. In the mobile application, measure v8.2b on the real iPhone, close up and far, with the 512-cluster basis.
9. Draw the channels of §4.11 for many assets, skies and cameras, aligned with the relit frames, and learn from them to
   predict the transport channels or the relit light from inputs that do not carry them -- a learned adapter on an
   image model that drives the light of a capture or a render as the bake would.

::: {.nota}
**The order** — The list follows the size of the measured defect and how direct the remedy is. Items 1–4 address the defects of §9.1 with remedies already quantified (least-squares curvature reduces the normal's $p_{90}$ error from 46° to 9.8°; 32 × 32 cells would halve the open class's excess at +96 B per Gaussian). Item 5 closes the validation gaps of §9.6. Items 6–8 are about scope (animation, levels of detail, mobile).

**Item 9** — It proposes using the bake as a data generator: drawing the parameter channels of §4.11 (normals, roughness, open directions, transfer) aligned pixel by pixel with relit frames, for many assets, skies and cameras, and training with them an adapter on a pretrained image model, so that the model predicts light transport from an image that does not carry it (a capture or an unbaked render). It is the same idea as control adapters on diffusion models (ControlNet, Zhang, Rao and Agrawala, 2023) and as recent work on relighting with diffusion models (for example DiLightNet, Zeng et al., 2024, or Neural Gaffer, Jin et al., 2024), with the difference that here the training pairs would be physically exact by construction.

*References:* §4.11 of the paper; L. Zhang, A. Rao, M. Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models", ICCV 2023; C. Zeng et al., "DiLightNet: Fine-grained Lighting Control for Diffusion-based Image Generation", SIGGRAPH 2024; H. Jin et al., "Neural Gaffer: Relighting Any Object via Diffusion", NeurIPS 2024. Code: the `render` tool of the MCP server (`modules/mcp`); `docs/decisions.md` (section "A transfer the stage turns, and the gaussians' parameters drawn (19 Oct)").
:::
