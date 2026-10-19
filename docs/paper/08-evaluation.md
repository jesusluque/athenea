<!-- Copyright (c) 2026 jesus luque. -->

# 8. Evaluation

This section gathers the measurements of Sections 2–7 into one account, by process. Part I, the bake, leads: the master
validated by relighting the baked Gaussians against the path trace of the source mesh (§8.2, §8.3), the engine (§8.4),
the levels of detail (§8.5) and the bake's costs (§8.6). Part II, the web format, comes last: its compression, its
behaviour on the phone and its memory (§8.7–§8.9), measured as a loss against the master. It introduces no new
numbers: every figure is quoted from the section named beside it. Where two sources give different figures for the
same quantity, §8.10 says which is which.

## 8.1 Setting

Desktop measurements were taken on an Apple M5 Pro under Metal; Vulkan and CUDA results come from an NVIDIA L4
(§2.15). The Corvette's reference is the mesh path trace from `/World/Camera` at 1920 × 1080 (§1.6). The player was
measured in three ways that are not interchangeable (§7): GPU timestamps from headless Dawn on the M5 Pro, the live
site in desktop Chrome in its mobile mode at 393 × 852 (the *emulated iPhone*, about twice as fast as the real one),
and a real iPhone (iOS 18.7, Safari 26.6.1, 393 × 852 @3).

## 8.2 The conversion against the path trace

| asset or test | measure | result | § |
|---|---|---|---|
| unit quad lit head-on (mesh pixel 0.5032) | converted pixel at $\sigma$ = 1.0 / material values passed | 0.456 / 0.5009 | 3.4, 3.6 |
| Corvette, 640 px, edges held vs control | relMSE / PSNR / SSIM, day frame | 0.0887 / 21.93 dB / 0.8567 against 0.0902 / 21.86 dB / 0.8554 | 3.5 |
| Corvette shadow catcher vs path trace | under the car / bumper contact / open ground | 0.192 / 0.030 / 0.220 against 0.182 / 0.043 / 0.217 | 3.9 |
| Corvette, 640 → 1280 px bake | car crop / frame | 16.55 → 17.88 dB / 22.18 → 23.90 dB | 3.11 |
| white pawn, body / glass head (2048²) | relMSE | 0.0218 / 0.1418 (later runs: glass 0.149, 0.081, 0.043) | 5.11 |
| sparrow (web conversion) vs mesh path trace | relMSE, raster | 0.0025 | 5.10 |
| sparrow wing, raster / traced | relMSE after coverage follows opacity | 0.1635 / 0.1247 | 5.10 |

## 8.3 The whole car along the pipeline

The same frame, relit under the path trace's sky and measured as each step of the bake and the draw was added, in
the order they were added. Rows from §4.10.3 and §6.10; nodes are drawn Gaussians.

| bake | configuration | nodes | car crop (dB) | frame (dB) |
|---|---|---|---|---|
| 640 px | all leaves, before §4's later steps | 5.38 M | 16.37 | 21.92 |
| 640 px | same bake, error cut 1 px, both culls | 675 760 | 16.28 | 21.80 |
| 640 px | error cut 0.25 px, back culling | 2 313 541 | 16.29 | 21.81 |
| 640 px | leaves, sharp lobes unwidened | 5.38 M | 16.58 | 22.15 |
| 640 px | cut, + coat held at folds | — | 16.51 | 22.15 |
| 640 px | cut, + polish field on rough materials | — | 16.55 | 22.18 |
| 640 px | leaves (0.01 px), no cull / cull | 4.35 M / 3.70 M | 16.73 / 16.75 | 22.40 / 22.43 |
| 640 px | group 0 with spill $w_s$ = 2 | 561 242 (group 0) | 16.79 | 22.38 |
| 640 px | turn capped / coat fold on paint only | — | 16.57 / 16.57 | 22.20 / 22.20 |
| 1280 px | cut 0.25 px, back culling | 4 998 976 | **17.88** | **23.90** |
| 1280 px | without the cap / gated sharp polish | — | 17.88 / 17.88 | 23.91 / 23.90 |
| 1280 px | gate + tube/fold split | — | 17.90 | 23.93 |
| 1280 px | tube/fold split (retained) | — | 17.88 | 23.91 |

Three conclusions follow. The level of detail costs about 0.1 dB against the path trace (all 640 px leaves against
the 1 px cut with both culls, 16.37 → 16.28 at an eighth of the leaves; §6.10). The shading steps of §4 moved the
640 px car by about 0.2–0.4 dB. The bake's resolution moved it by 1.3 dB on the crop and 1.7 dB on the frame (the
640 px cut with the polish field against the 1280 px cut). PSNR over a crop moves little, so the class ratios below
show where the changes act.

| region or class (mean over the path trace's) | before | after | condition | § |
|---|---|---|---|---|
| door, mirror to ground < 3 m | 0.78 | 1.02 | 640 px leaves → 640 px cut with polish field | 4.10.4 |
| door, mirror to ground 3–30 m | 0.69 | 0.88 | 640 px leaves → 640 px cut with polish field | 4.10.4 |
| door, mirror to the HDRI (open) | 0.92 | 1.13 | 640 px leaves → 640 px cut with polish field | 4.10.4 |
| door joints | 0.69 | 1.04 | 640 px leaves → 640 px cut with polish field | 4.10.4 |
| door open reflections, local contrast ($\sigma$ 1.5 / 4 px) | 0.35 / 0.53 | 0.56 / 0.74 | 640 px cut with polish field → 1280 px cut | 4.10.4 |
| hood highlights, offset $p_{50}$ | 3.2 px | 2.0 px | 640 px cut with polish field → 1280 px cut | 4.10.5 |
| hood mean | 0.89 | 0.98 | 640 → 1280 px | 4.10.5 |
| grille's holes in the coarse cut | 2.24 | 1.04 | 640 → 1280 px | 3.11, 6.5 |
| sill's hairs (pixels) | 987 | 459 | 640 → 1280 px | 3.11 |
| sill, under the car (phone) | 1.66 | 1.08 | without → with the contact shadow | 7.9 |
| chrome strip high-pass / PT | 0.41 | 0.41 | 1280 px cut → tube/fold split (gated sharp polish: 1.12) | 4.10.6 |
| bright specks on the sill's fold | 10 | 1 | 1280 px cut → tube/fold split | 4.10.6 |

| material (relMSE against the path trace) | chrome | rough metal | paint | § |
|---|---|---|---|---|
| first transfer | 0.536 | 0.0896 | 0.705 | 4.10.1 |
| TX, cells and field everywhere | 1.64 | 0.221 | 0.393 | 4.10.1 |
| TX with the polish gate | 0.591 | 0.0835 | 0.341 | 4.10.1 |
| + sharp coat and sharp polish per pixel | 0.545 | 0.083 | 0.281 | 4.10.1 |

## 8.4 The engine's correctness and speed

Every check below is a GPU kernel against a closed form, an independent estimator, MaterialX, or Storm.

| check | result | § |
|---|---|---|
| two half-opaque discs, raster / reference / ray tracer vs analytic over-operator | p99 0 (encoded blend: p99 26) | 2.4 |
| converted planes of coverage 0.25 / 0.5 / 0.75, raster at full size | 0.251 / 0.501 / 0.752 | 2.5.3 |
| shutter step 0.03 vs N instants | p99 2, energy ratio 1.0000 | 2.5.3 |
| ray tracer vs its exact-order reference | p99 ≤ 1, at most 0.05 % of pixels over 2 | 2.15 |
| mesh visibility vs Storm (Kitchen_set), raster / rays / bvh | segmentation 18 / 41 / 36 differing pixels of ~57 k | 2.8 |
| MaterialX graphs vs MaterialX's own genglsl, 65 536 directions | worst 3.2·10⁻⁵ over eleven graphs | 2.9 |
| lights vs closed forms; χ² against `lightPdf` | 0 of 8281 pixels beyond 2 %; $\lvert z\rvert \le 1.61$ | 2.10 |
| path tracer: closed emissive shell; convergence exponent | 0 of 3072 beyond 10⁻⁴; −0.499 | 2.11 |
| OIDN, 16 paths vs 4096-path reference | relMSE 3.65·10⁻⁴ → 7.17·10⁻⁵ | 2.11 |
| test suite on three devices (188 cases) | Metal 188 passed; Vulkan 180 passed, 8 skipped; CUDA 96 passed, 1 failed, 91 skipped | 2.15 |
| a TX cloud rotated 60° with its camera and sky vs unrotated | pawn relMSE 9·10⁻⁶ (61 dB); test balls 1–2·10⁻⁴, 1.5·10⁻²–0.13 under a sky left unturned | 4.3 |
| OpenFX node vs the server's own render, same camera | relMSE 0; in Natron, colour and world position within 10⁻⁶ of a plain OpenFX host | 2.14 |

| workload | figure | § |
|---|---|---|
| radix sort, 10 M 32-bit pairs | 12.4 ms (39.9 ms before) | 2.3 |
| sparrow clips, 5 887 323 gaussians at 1600 × 1000, playing | 102 → 49 ms a frame | 2.5.3 |
| train_7k (742 k), 1080p: tile raster / ComputeBvh / hardware tracer | 13 / 343 / 435 ms | 2.6 |
| Kitchen_set visibility at 1920 × 1080: raster / rays / bvh | 31.7 / 19.2 / 29.2 ms | 2.8 |
| Kitchen_set lit at 1920 × 1080: raster / path tracer (1 path, 1 bounce) | 96.0 / 358.9 ms | 2.11 |
| whole TX Corvette (14.7 M gaussians) at 1920 × 1080, desktop rasteriser | 158 ms (about 83 ms for a relit cloud of its size) | 4.10.2 |
| relightable Corvette (3 M) through the OpenFX node in Natron, 1920 × 1080, every plane | first frame 7.5 s (stage opened); then 120–170 ms, 190–220 ms with the sky turning | 2.14 |
| one parameter channel, pawn at 768 × 768 | 20–60 ms | 4.11 |

## 8.5 Levels of detail

| level | leaves | merges | roots | pages | occlusion codebook | cells reopened | § |
|---|---|---|---|---|---|---|---|
| 640 px car | 5.38 M | — | 17 133 | 3 582 | 8192 | 11.9 % | 6.2.5, 6.7 |
| 1280 px car | 8.69 M | 8.67 M | 20 723 | 5 789 | 8187 | 12.3 % | 6.7, 6.10 |
| 5120 px close-up level | 74 536 126 | 74 328 031 | 208 095 | 49 634 | 8 182 | 0–23.6 % per tree | 6.7, 6.9 |

The close-up level is 24 trees and 148 864 157 nodes, 87.4 GB on disk, with its shared occlusion written in place
(645 s, 10.2 GB peak resident; §6.9).

| cut, at 393 px wide, against the leaves at 4× | near nodes | near dB | far nodes | far dB | § |
|---|---|---|---|---|---|
| error 1 px, bounds | 666 585 | 33.29 | 343 541 | 39.15 | 6.10 |
| + back culling of closed shells | 522 931 | 33.63 | — | — | 6.10 |
| + occlusion | 360 333 | 33.58 | 195 049 | 39.53 | 6.10 |
| both culls, $O<2\sqrt{E}$ rule | 378 028 | 33.53 | — | — | 6.10 |

Each cull lowers the node count, and no culled cut reads lower against the leaves than the error cut alone (§6.10). In the player, every cull on draws 51 % fewer nodes than every cull
off, at 36.9 dB between the two images, and the occlusion codebook reaches 84 % of the ideal (§7.5.1). Residency is the
weak point: a walk needs about 2.2× the cut, and pages hold 3.6–4.5× it whatever their size (§6.8), which is why the
player keeps the traversal apart from the payload (§7.5.2).

## 8.6 Costs

| step | figure | § |
|---|---|---|
| 1280 px car, 8 groups, transfer each | 212–304 s, about 34 min in all | 3.11 |
| shadow catcher, 309 k gaussians | 79 s | 3.9 |
| sparrow's per-part visibility fields (12 parts) | 22 s | 5.8 |
| sparrow decimation keeping the rig (5.89 M) | 28 s | 5.9 |
| shared occlusion codebook over the close-up level | 645 s | 6.9 |
| original LOD build, train_7k | 47 ms | 6.1 |

## 8.7 Compression

| step | before | after | image cost | § |
|---|---|---|---|---|
| byte planes (lossless), Corvette light / HD | 68.1 / 288.7 MB | 60.9 / 248.6 MB | none | 7.1 |
| transfer encoding 3 (CPCA in the file), Corvette light set / pawn page | 67.9 / 281 MB | 49.5 / 84 MB | per splat relMSE $4\times10^{-8}$–$3\times10^{-5}$ | 7.2.1 |
| joint CPCA on the GPU, whole-car basis (256 clusters) | 112 values | 1 id + 24 int4 weights | transfer relMSE 1.03·10⁻²; image within relMSE 10⁻³ | 7.2.2 |
| 512-cluster basis (v8 trial), fine3_0 | 0.0181 | 0.0136 | same shade time on the M5 Pro | 7.2.2 |
| open-direction bits by VQ, 1024 centroids a piece | 256 bits | u16 index | 62.9–65.4 dB | 7.3 |
| compact record (resident) | 308 B a gaussian + 29 MiB | 42.4 B + 16 B a drawn splat + ~3 MiB | 40.2–41.9 dB against the full relight | 7.4 |
| traversal record | 24 B (Spark) | 5–11 B by class | same cut | 7.5.3 |

## 8.8 The phone

**Against the path trace, by data version** (validate-corvette regions, PSNR in dB at 1920 × 1080; §7.12.3):

| version | car | door | sill | bumper | hood |
|---|---:|---:|---:|---:|---:|
| v6 (640 px trees) | 19.33 | 33.77 | 31.87 | 20.21 | 14.16 |
| v7 + pixel spread, $\tau$ 1 (1280 px) | — | 35.53 | 32.36 | relMSE 0.199 | 14.46 |
| v7, current defaults | 19.03 | 35.52 | 32.36 | 22.28 | 14.68 |
| v8 (5120 px close-up level) | — | — | — | — | — |

v8 had not been converted for the player at the time of writing and has not been measured. On the car crop, the
compact path's per-pixel sharp lobe took the car from 16.49 to 17.46 dB (§7.12.3).

**Speed** (§7.12.1, §7.12.2):

| device | data | drawn | result |
|---|---|---|---|
| real iPhone, stress page | plain 3DGS | 200 k / ~1 M | 58.8 / 26.3–27 fps |
| real iPhone, stress page | compact, real basis | 200 k / 400 k / ~1 M | 27 / 47 / 108 ms |
| real iPhone, model | compact | — | ~115 ms a million + 4 ms: ~250 k at 30 fps, ~110 k at 60 fps |
| real iPhone, first device test | v7, default view | 106 k | 53 fps ($\tau$ 8.00) |
| real iPhone, first device test | v7, close up | 160 k | 32 fps, soft logo and grille |
| emulated iPhone | v5 / v7 / v7 after a later frame-time adjustment | 226 k / 216 k / 234 k | 60 fps at $\tau$ 3 / 2.22 / 1.95 |
| emulated iPhone, still frame | v7, close up | 529 k | one frame at $\tau$ 1.23 |
| Dawn, M5 Pro, compact pass | 393 × 852 $\tau$ 3 / 1920 $\tau$ 1 | — | 4.0 / 17.2 ms |

By the model, the far LOD cut with both culls (195 049 nodes, §8.5) fits the ~250 k the phone draws at 30 fps, and the
near cut is about 1.4× that (§6.10). A second device test found the close-up view still soft. The softness was traced
to the 1280 px cloud, not to the player, which led to the 5120 px close-up level (§7.12.1, §6.9). For the sparrow, the
mobile cut (488 656 splats, 31.7 MB) ran at 60 fps headless in mobile mode, animated (§5.12).

## 8.9 Memory

| quantity | figure | § |
|---|---|---|
| car data, v6 after the atlas, $\tau$ 3: Camera / close / far | 39.0 / 43.8 / 33.0 MB | 7.11 |
| render state, same views | 24.1 / 26.0 / 21.7 MB | 7.11 |
| page (JS heap, three.js, domes) | ~39 → 25 → ~17 MB | 7.11 |
| car data, v7, $\tau$ 3 | 31–36 MB | 7.11 |
| car data, still frame close up (budget ~100 MB) | 95.6 MB | 7.10, 7.11 |
| per gaussian, earlier relight / compact path | 308 B + 29 MiB / 42.4 B + 16 B a drawn splat + ~3 MiB | 7.4, 7.11 |
| at 1 M splats, earlier / compact | 326 / 58.6 MiB | 7.4 |
| tree write, 4.08 M-leaf door (after the fix) | 10.5 GB peak resident | 6.9 |
| shared occlusion over 24 trees | 10.2 GB peak resident | 6.9 |

The player's 50–60 MB target applies to car data only; render state is counted apart (§7.11). Moving, the car data
was 49.1–59.0 MB in v6 before the atlas and 31–44 MB in v6 after it and in v7, inside the target (§7.11).

## 8.10 Where sources differ

- **1280 px leaves.** The eight bakes wrote 8 703 429 Gaussians (§3.11, "8.70 M"); the trees hold 8.69 M leaves
  (§6.10). The first is the conversion's count, the second the trees'.
- **Front_Bumper_001.** The transfer bake reports 5 009 059 Gaussians, the tree 5 008 986 leaves (§3.11); the totals
  use the tree's figure.
- **The close-up level's size on disk.** 87.4 GB for the 24 trees; "about 86 GB" was the estimate before the last tree
  was written (§6.9).
- **The close-up occlusion total.** A first count read 132.4 % reopened; it was a 32-bit overflow of the closed-cell
  total, since fixed. The per-tree figures (0–23.6 %) and the codebook were correct (§6.7).
- **The still frame.** One measurement gives 529 k drawn at $\tau$ 1.23 with 95.6 MB of car data; another gives 531 k
  at $\tau$ 1.22 in 118 ms (§7.10). They are two runs of the same policy.
- **Door joints at 1280 px.** §6.10 gives the joints as rising from 0.90 to 1.02 between the 640 px cut with the
  polish field and the 1280 px cut, whereas the class table of §4.10.4 gives the former's joints as 1.04 (0.90 is the
  640 px leaves without culling). The two figures come from different measurement runs; both put the 1280 px joints
  within 2–4 % of the path trace.
- **Car crop on the phone.** 17.46 dB for the compact path (§7.12.3) and 17.88 dB for the 1280 px cut on the desktop
  rasteriser (§4.10.3) are separate measurements on different paths and data, and are not compared here.
- **Real versus emulated speed.** Emulated figures run at about twice the real iPhone's; draw counts are judged on the
  device (§7).
