<!-- Copyright (c) 2026 jesus luque. -->

# The Corvette on an iPhone: what was applied (11-12 Oct 2026)

The whole Corvette, transfer included, drawn on an iPhone and judged against the mesh path trace
(`athenea-renders/simplify3/pt/day_pt_Camera.exr`, `/World/Camera`, 1920 x 1080), on the whole frame and on crops of
the small pieces. This document records what was applied in `athenea-pieces` (branch `pieces`) and what was decided
with the web player (sparkwebGPU, session `sparkwebgpu-ce`), with what was measured. The running log is
`PIECES-PLAN.md`; the Spanish translation of this document is `docs/pieces.es.md`.

Figures marked "car crop" are PSNR on the 900 x 420 crop of the car (`--cut 900x420+200+420`) against the path
trace; "frame" is the whole 1920 x 1080 frame. Ratios such as 1.04 are a region's mean over the path trace's
(research's boxes, `athenea-research/proto/door/*.py`).

## 1. Shading

### 1.1 Sharp lobes read at the roughness before the bake widened it (b26d276)

The bake widens a gaussian's stored roughness by the spread of normals under it (Toksvig, `--specular-filter`):
alpha'^2 = alpha^2 + 2 s^2, with s = kMax max(scale) from the shape operator. The per-pixel sharp lobes
(`SplatSlope`) already turn the normal across the gaussian, so reading them at the widened roughness counted the
spread twice. `unwidened(r, s) = (r^4 - 2 s^2)^(1/4)` (held at 0) undoes it for the coat's and the polish's sharp
lobes; `slope.bakeSpread` carries s.

Research (t64): the door's open-reflection contrast went from 0.12/0.26 to 0.40/0.61 of the path trace's, the
hood's from 0.27 to 0.41; the darkness in the ground class (0.67-0.78) and at joints (0.69) stayed.

### 1.2 The polish sees its ground (35ae3cb)

Ported from `athenea` (tx-transfer-complete, 4c7dc40): the cloud carries the material's own roughness,
`primvars:athenea:splat:materialRoughness` (before `--specular-filter`; -1 under a map). Where it is at least
`kPolishFieldRoughness` (0.3) the polish asks its own lobe's cells, and the reflected field coupled to the sky answers
what they close; below it, or without the array, as before. Paint is 0.34; chrome 0 and rough metal 0.22 keep the
closed part dark.

The value rides on the pbr word: bit 26 says it is there, bits 27-31 hold it in 5 bits
(`kMaterialRoughnessMark`, 8.0 per step on the transmission). The streams kernel already binds every buffer Metal
gives a kernel, so the mark is added after it in a pass of its own, `splatMaterialRoughness` in
`scene/splat_hardness.slang`, beside the hardness's.

### 1.3 A fold's coat held to 0.3, on paint only (35ae3cb, 5d0c530)

At a fold the bake widened the lacquer to near 1, and a joint drew a dark streak. `kCoatFoldRoughness` (0.3) holds
it: coat = max(unwidened(c, s), min(c, 0.3)). Measured by region in the web player, it lit chrome, glass and a rim's
spokes the path trace shows dark (chrome -1.4 dB, spokes -0.5 dB), so it applies only where the material's own
roughness is at least `kPolishFieldRoughness`. At 1280 px the bake no longer widens the coat (it comes out at 0), so
this term changes nothing on the re-baked car.

### 1.4 A tube turns, a fold widens (f9c9a2b)

The per-pixel turn extrapolates a gaussian's normal from its centre triangle's shape operator; past an edge it no
longer holds. The stored normal and curvature are exact at the centre (research, hoodnormal: p90 0.009 degrees,
correlation 1.000); 64 % of a hood's footprints cross an edge.

t, the turn across +-2 sigma, is twice the bake's spread. Research (tubecrease) found it separates the two cases in
the 1280 px cloud: a chrome tube's body sits at t 0.3-1, coherent with its neighbours; normals smoothed across a fold
within triangles narrower than the gaussian sit past 1 (a sill's 5-9 rad).

| t | the pixel's normal | the sharp lobes |
|---|---|---|
| under `kFoldTurnLow` (0.7) | turns in full | as stored |
| between | smoothstep | smoothstep |
| past `kFoldTurnHigh` (1.5) | does not turn | widened: alpha' = sqrt(alpha^2 + (t/2)^2), held under `kFoldRoughness` (0.5) (`foldWidened`) |

### 1.5 What was tried and left out

| change | test | result |
|---|---|---|
| the turn held to 0.2 rad over the footprint (`kMaxFootprintTurn`) | T69 | car crop 16.55 -> 16.57; the hood did not move (3.16 px); the front rim's joints 1.48 -> 1.54. Replaced by 1.4 |
| the sharp polish gated on the unwidened roughness | T73 | the chrome strip's high-pass error tripled (0.41 -> 1.12 of the path trace's) |
| a coat floor, or footprint 0, on the phone | CW | the door's blotches did not change: they were the cloud's |
| `--specular-filter 0` | bake test B, C | a dashed dark line on the sill's fold; the chrome strip sparkles. The polish needs its widening there |
| `--edge-reach 1.0` | bake test C | elongation and hairs identical to B |

### 1.6 Results against the path trace

| test | what | car crop | frame |
|---|---|---|---|
| T63c | 640 px, cut 0.25 px with back culling | 16.29 | -- |
| T65 cut_a | + 1.1 and 1.3 | 16.51 | 22.15 |
| T65 cut_b | + 1.2 | 16.55 | 22.18 |
| T66 | leaves (cut 0.01 px), no cull / cull | 16.73 / 16.75 | 22.40 / 22.43 |
| T69 / T70 | + the 0.2 rad cap / + coat on paint only | 16.57 / 16.57 | 22.20 / 22.20 |
| T71 | **1280 px bake**, cut 0.25 px with back culling (5.0M) | **17.88** | **23.90** |
| T72 | T71 without the cap | 17.88 | 23.91 |
| T73 | T71 + the gate of 1.5 | 17.88 | 23.90 |
| T74 | T73 + 1.4 | 17.90 | 23.93 |
| T75 | T71 + 1.4 (committed) | 17.88 | 23.91 |

T66 (research t66): back culling is right. It lifts the whole door about 12 % evenly (door 1.01, near ground 1.02 of
the path trace); the door's open reflections at 1.13 are the coat letting through sky where the mirror meets the car
itself (16 x 16 cells of 11 degrees). T71 against cut_b (research): hood highlights 3.2 -> 2.0 px off, mean 0.89 ->
0.98, joints 0.90 -> 1.02; the door's open-reflection contrast 0.35/0.53 -> 0.56/0.74; the grille's holes in the cut
2.24 -> 1.04. T75 against T71: bright specks on the sill's fold 10 -> 1, the door 405 -> 389, the rim 204 -> 190.

## 2. Decimate and the merge tree

### 2.1 New options

`--mark-material-roughness`, `--cut-spill`, `--cut-spill-floor` (see `docs/operations.md`, 2.4.1). `--tree-coverage`
now also writes `primvars:athenea:splat:lodSpill`.

### 2.2 A merge's spill (1bb79f4)

E = sum max(a G - C, 0) / sum a G over the coverage fit's own grid (8 x 8 over +-1.25 sigma on the node's plane, C the
leaves' composite, a the fitted coverage): the share of what a merge paints that its leaves do not. It matches
research's numpy to a correlation of 0.994. The cut adds r w_s max(E - E0, 0) to a node's error.

T67 on the grille's group: w_s 2 fills its holes no more than the leaves (1.90 -> 1.09 of them; car crop 16.55 ->
16.79) but holds 82-99 % more nodes. Research (spill_*.py): every merge spills about 0.16, and the two-leaf merges
of Chrome_Part_Grill's sub-pixel strips that fill the holes spill no more than a panel's; no floor separates them,
and no view-independent measure does. **Default 0.** At 1280 px the holes are at 1.04 without it.

### 2.3 Trees of millions of leaves (5ed0b6a)

A 5120 px wheel (4.45M leaves, 8.88M nodes) built its tree and ran out of device memory writing it.

- `mergeAttribute` allocated a weights buffer as large as each attribute, though only a rig's joints (mode 3) write
  it: 1.7 GB beside each of a transfer's 48-float arrays. It is one word elsewhere.
- decimate's write held the arrays three times over in the memory the GPU shares: a copy of what was read, and the
  merged arrays started as another copy. They now start from the metadata alone, and the tree's write, the last,
  releases each array once it is on the device.

Measured after: a 4.08M-leaf door, 10.5 GB peak resident; 4.90M passed, 5.01M did not at first (section 3.4);
a later run wrote it, with 2.23 GB as the lowest free memory.

### 2.4 Occlusion written over each tree

`occlusionMasksShared` writes a tree whose output is its input beside it (`<tree>.occlusion.usdc`) and renames it
over it: the close-up level's trees are 87.4 GB together, and a copy of each would not fit beside them. Run over the
24 trees of car5120: one shared codebook of 8,182 entries in 645 s, 10.2 GB peak resident; per tree 0-23.6 % of the
closed cells reopened (Windshield 0.0 %, pan2 23.6 %). The first total read "132.4 %": the closed-cell total over
~150M nodes wrapped a 32-bit counter; the per-tree figures and the codebook were right, and the totals are now summed
in 64 bits (4a8f8c2).

## 3. Bakes

### 3.1 The whole car at 1280 px (car1280)

The 640 px bake's leaves were 3-4 mm on the hood; the door's stripes and the sill's hairs were in the cloud (the
leaves showed them; the path trace did not). The car was re-baked at `--camera-pixels 1280` with the 640 bake's
settings, in 8 groups of top-level parts balanced by splats. Each group is converted with every other part
`--hide`n: `--hide` reaches the conversion only, the bake traces the whole stage, so a part keeps the others'
shadows. Part names come from the cryptomatte manifest (the log's interleaved lines garbled some).

| | 640 px | 1280 px |
|---|---|---|
| leaves | 5.38M | 8.70M |
| a panel (hood / door) | 50k / 79k | 151k / 266k |
| glass, tyres | one a triangle | one a triangle (barely grow) |
| occlusion 8192 shared, cells reopened | 11.9 % | 12.3 % |
| T71 cut 0.25 px: car crop / frame | 16.55 / 22.18 | 17.88 / 23.90 |

The pieces build writes `closed` and `materialRoughness` a gaussian, each material of a mesh its own.

### 3.2 The bake test (8 parts at 1280 px)

Door, Door_001, Panel_under_Door, Panel_under_Door_001, Front_Fender, Chrome_Part_Grill, Grill_Rim, Hood; 931,017
gaussians each. Research (attr.py): at 1280 px the coat's roughness comes out at exactly 0 (the 640 bake had it at
0.17-0.24 with 5-14 % high-pass on folds); the sill's hairs fall 987 -> 459 px from the resolution alone.

| variant | settings | result |
|---|---|---|
| **A** | as the 640 bake | **chosen**: no hairs, the door smooth |
| B | `--specular-filter 0` | dashed line on the sill's fold, chrome sparkles |
| C | B + `--edge-reach 1.0` | as B |

### 3.3 The close-up level (car5120)

On the iPhone close up, the logo and grille stayed soft: the camera was far nearer than `/World/Camera`, with about
0.5 mm a phone pixel against 3-4 mm leaves. A logo-and-grille test at 1280 and 5120 px from a close camera showed the
5120 px leaves crisp (the checkered flag, the bowtie, the grille's mesh); 108,527 -> 404,594 gaussians.

- **Detailed parts**: the 66 parts the 1280 px bake already gave at most 1.5 gaussians a triangle (badges, grilles,
  lamps, calipers, mirrors, wheels, wipers, trims), re-baked at 5120 px in 7 groups.
- **Panels**: the user asked for the large pieces too. Each part at 5120 px, or as fine as keeps it under a cap of
  leaves (5M, then 4.5M after a 5.01M tree did not fit), the small ones grouped up to 4M.
- `--max-cells 1048576`: at 4736 px, Infinite_Black lost 359,628 cells past the default 262,144 a triangle (holes);
  re-baked whole.

| groups | leaves | resolution |
|---|---|---|
| fine0-6 (66 detailed parts) | 18,610,627 | 5120 |
| door (pilot) | 4,084,170 | 5120 |
| Bottom | 4,897,428 | 3392 |
| Part_3 | 4,370,459 | 4480 |
| Infinite_Black | 4,481,738 | 4736 |
| Panel_under_Door_001, Windshield, Front_Fender, Hood, Front_Bumper, Part_2, Part_23 | 15,994,566 | 5120 |
| pan0-4 (small panels) | 17,088,152 | 5120 |
| Front_Bumper_001 | 5,008,986 | 4480 |
| **24 trees** | **74,536,126** | |

The groups' merge trees are per part, so the close-up level is the same format as car1280: the coarse roots stay,
and the deeper levels are only paged in close up. Each group takes the shared GPU lock on its own. The 24 trees hold
74,536,126 leaves, 74,328,031 merges and 208,095 roots: 148,864,157 nodes in 49,634 pages, 87.4 GB on disk, with the
shared occlusion written in place (2.4).

### 3.4 Limits met, and how

| limit | where | handled by |
|---|---|---|
| disk (5 GB free at one point) | the 1280 and 5120 bakes | old intermediates deleted (with the user's OK); a group's bake deleted once its tree is written; occlusion in place (2.4) |
| device memory writing a tree | 4.45M-leaf wheel | 5ed0b6a (2.3) |
| device memory past ~5M leaves | Front_Bumper_001 (5.01M) | a per-part cap of 4.5M leaves |
| cells a triangle | Infinite_Black | `--max-cells 1048576` |
| a 1280 px group list from a garbled log | car1280 | part names from the cryptomatte manifest |

## 4. The phone (sparkwebGPU; decisions, its code lives there)

| what | decided because |
|---|---|
| a contact shadow from the shadow catcher (`pieces/shadow/shadow.usdc`, 309k, merged 2 x 2 to 78k, 2.5 MB of GPU) | the car floated; the sill's x1.6 was the hall's floor at infinity seen through the car's edge. Sill 1.66 -> 1.08 of the path trace (desktop 1.07). Pure 3DGS: gaussians only, no mesh, no AO |
| the per-pixel sharp lobe in the compact path, with the pixel spread alpha^2 += (\|dx\|^2 + \|dy\|^2)/6 | car crop 16.49 -> 17.46 dB (the desktop's); without the pixel spread v7 had bright specks on folds (27/25 -> 2/1) |
| the environment as a filterable rgba16float atlas | the sharp fragment 11.8 -> 4.0 ms on the phone |
| 512 transfer clusters (the 9th bit in the shadow index's free bit 15) | one file of the 5120 detail at 0.0181 relMSE; with 512 at 0.0136, every other group 20-30 % better, the same shade time |
| a multi-level synthetic root in the packer | a file holds about 3000 roots; the 5120 trees have thousands of lone roots (merges never cross parts or guards, on purpose) |
| v7 = car1280 | door 33.77 -> 35.53 dB, sill 31.87 -> 32.36, hood 14.16 -> 14.46 |
| draw target by frame time; quality over frame rate (user: "aunque empeore el rendimiento quiero calidad") | the real iPhone settled at 106k drawn, tau 8, 53 fps; close up 160k, tau 8, 32 fps |
| a still frame: after 300 ms still, one frame at tau ~0.5-1.2 with no time limit, nothing more until a touch | close up 531k at tau 1.22 in 118 ms (emulated at 190 ms a million); the first frame after a touch 39-47 ms |
| car data ~100 MB at rest, ~55 MB moving (user) | the still frame's finer pages |
| v8 = car5120 (the close-up level) | its 24 trees and shared occlusion are written; the player's conversion was in progress |

Figures from the emulated phone are about twice the real iPhone's; draw counts are judged on the device.

## 5. Operations

GPU jobs run one at a time on the machine through `~/luc/athenea-gpu/shared.sh`, which takes a lock directory shared
with the web player's jobs (`mkdir`), then runs the job under `guard.sh` (it stops the job on thermal pressure,
SIGSTOP and SIGCONT). Each step runs under `memguard.sh` (it kills the job when free memory falls under 1.5 GB). A
job waits 25 s before asking for the lock, so the jobs that poll every 10 s get in between, and waits while
`~/luc/athenea-gpu/hold` exists (a long gap given to another session). Another session's short GPU need is met by
pausing the running bake (SIGSTOP on the binary and its `memguard.sh`, SIGCONT after). A build directory a running or
queued job uses is not rebuilt.

## 6. Not done

- v8 (car5120) in the player: its conversion, then its quality against the path trace and its behaviour on the real
  iPhone.
- Lamps: the bake has no lamp emission (the headlights are dark against the path trace).
- The chrome strip on the front bumper: 38 % brighter than the path trace, its peak lower (a tube a few pixels
  across, each gaussian reading its mirror).
- The rim's near-ground reflection at 0.72 of the path trace.
- The door's open reflections at 1.13: the coat lets sky through 16 x 16 cells of 11 degrees where the mirror meets
  the car; 32 x 32 cells (+96 B a gaussian) would halve it.
- The front bumper's lower lip, ragged; the hood highlights 2.0 px off (a curvature fit over the footprint from the
  mesh would halve the median error).
- The grille's holes in a coarse cut: no per-node measure separates them; at 1280 px and finer they are at 1.04.
