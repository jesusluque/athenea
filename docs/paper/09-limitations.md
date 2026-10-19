<!-- Copyright (c) 2026 jesus luque. -->

# 9. Limitations and future work

The limitations below are gathered from the sections' own lists (§2.16, §3.12, §4.12, §5.13, §6.11, §7.13) and
grouped by where they act, the relit bake first and the mobile application last. Each is stated with the figure that
measures it, where one exists.

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

## 9.5 Levels of detail, streaming and the player

- **Occlusion weighting.** The shared codebook is apportioned by entries used, not by how often a view culls a node;
  weighting by an orbit of views in the bake should lift culling past 84 % of the ideal (§6.11).
- **Not ported.** The closed-shell back rule and the error cut live in `athenea-pieces`; the main engine's
  `CutSelector` is unchanged (§6.11).
- **Residency.** Pages hold about 4× the cut because needed and view-culled nodes share every page; the player splits
  traversal from payload rather than shrinking pages (§6.8, §6.11).
- **Merge opacity in the player.** The player draws each merge's mass, while athenea's rasteriser keeps the
  least-squares coverage; the coverage fit is the better choice except on the grille (§6.4).
- **The cut in the player.** The compact traversal does not keep a merge's LoD size, so a combined size/error cut has
  nothing to guard with, and the error cut alone did not beat the size cut in every view (§7.13).

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
