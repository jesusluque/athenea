<!-- Copyright (c) 2026 jesus luque. -->

# Part II. The web format: a derived delivery format {.unnumbered}

The second process starts where the first ends. It takes the master of Part I and derives from it a format for one
purpose: delivering the relit asset to a web browser on a phone. Its goal is different, and so is its criterion. A
phone tab has a memory budget (about 50–60 MB of asset data while the camera moves, about 100 MB at rest), a download
that must start drawing within seconds, and a frame time; the format is made to meet those, and its fidelity is
measured as a loss against the master's own uncompressed relight and, through it, against the path trace. Nothing in
this process changes the master, and nothing in it is read back by the bake.

::: {.nota}
**Master versus delivery format** — Part I produces a *master*: the highest-fidelity representation (USD with every gaussian, its transfer in f16, its 256 bits of open directions, its complete merge trees) from which anything can be rebuilt. Part II produces a *distribution format*, just as video distinguishes a ProRes/EXR master from an H.264/AV1 delivery file: the second is derived from the first, loses information irreversibly and is optimised for one particular receiver. Here the receiver is a mobile browser tab, and its three constraints are what fix the design:

- **Memory.** Safari on iOS does not publish a per-tab limit; WebKit and the system (iOS's *jetsam* mechanism) kill the web content process when its total footprint exceeds a threshold that depends on the model and on the system's memory pressure. The paper therefore sets its own conservative budget: 50–60 MB of *asset data* with the camera moving and about 100 MB at rest (§7.10–§7.11), kept apart from the render state and the rest of the page.
- **Start-up latency.** A page that takes tens of seconds to show anything gets abandoned; the format must allow a coarse cut to be drawn from very few bytes (each tree's page 0) and refined afterwards (§7.5–§7.6).
- **Frame time.** 16.7 ms for 60 fps, 33.3 ms for 30 fps. On an iPhone the shade cost per drawn gaussian dominates (§7.12.1), so the number of gaussians drawn per frame is the control variable (§7.10).

**The chain of fidelity measurements** — The web format's loss is measured against the *uncompressed* relight of the same master (which isolates the compression error), and the master is measured against the mesh path trace (which isolates the bake error). Since errors compound, a figure against the path trace mixes both; that is why the paper always states the reference of each number (§7, "Three kinds of measurement appear").

*References:* Part I of the paper (§2–§6); the sparkwebGPU player, with public demos at https://sparkwebgpu.lucab.co.uk ; the athenea engine, at github.com/jesusluque/athenea.
:::

**Input.** The master's merge trees with their pages and occlusion codebook (Section 6); for the Corvette, the 24 trees
of the close-up level.

::: {.nota}
**Merge tree, pages and occlusion codebook** — The three products of section 6 that this process consumes. A *merge tree* is a cloud's level-of-detail hierarchy: the leaves are the bake's gaussians and each interior node (a *merge*) is a gaussian that replaces its children, with an error bound $\varepsilon$ and a sphere of radius $\rho$ that bound how noticeable the substitution is. *Pages* are pieces of the tree (about 3000 nodes) ordered by error bands, the streaming unit (§7.5.1). The *occlusion codebook* is a dictionary of 16 × 16 direction masks (in octahedral projection) that says from which directions some part of each node's subtree is visible; each node stores an index into it (§7.5.1, "Occlusion").

**Close-up level and the Corvette's 24 trees** — The *close-up level* is the finest bake level (`car5120`, baked at a 5120 px camera, §6.2.1), split by groups of car parts: 24 independent trees. Each tree is one file (or several, §7.5.4) in the web format.

*References:* §6 of the paper; [35] (Nanite, the idea of paging by error); athenea `apps/athenea/src/CmdDecimate.cpp` (`--tree`, `--tree-pages`, `--occlusion-mask`), athenea `modules/usd/src/Occlusion.cpp`, athenea `shaders/athenea/lod/lod_pages.slang`; `docs/decisions.md` (section "Decimation: the levels of detail, cut once by what a merge loses").
:::

**The process**, in order, run by the player's tools rather than by the engine:

1. **Reading the master** (`usd-athc`, a reader built on the `openusd` crate): the USD cloud becomes an `.athc`
   version 3 container, laid out in sections and tiers so that one HTTP Range fetches what one tier of one block needs
   (§7.1). A skinned cloud adds its skin and its skeleton and clips (§5.12).
2. **Lossless packing** (§7.1): each array transposed into byte planes, or into planes of differences, whichever gzip
   compresses better, since gzip is what a browser decompresses.
3. **The transfer** (§7.2): either a clustered PCA of each block of the file (encoding 3), or a basis fitted over the
   whole asset (`fit-compact-basis`) that the phone keeps resident, each Gaussian then holding a cluster id and
   $3\times8$ int4 weights.
4. **The open directions** (§7.3): a Hamming-space dictionary of 4096 masks shared by the asset; each Gaussian holds a
   16-bit index.
5. **The compact record** (§7.4): every remaining parameter quantised into byte planes, 40 bytes a Gaussian on the GPU.
6. **Pages, tiers and traversal records** (§7.5): the tree's pages cut into error bands, traversal records of 5–11
   bytes, and an asset table (the bases, the dictionary, the materials) fetched once.
7. **At playback** (§7.6–§7.10): streaming by Range requests, a cut per frame within the memory budget, the draw and
   the relight on the phone, a contact shadow, and a still-frame policy that refines when the camera stops.

::: {.nota}
**`usd-athc` and the `openusd` crate** — `usd-athc` is a Rust binary of the player that reads the master's USDC without Pixar's C++ library: it uses `openusd` (version 0.7.0), a Rust implementation of a USDA/USDC reader. It is a practical decision: the player's tools build without the engine's OpenUSD/MaterialX/TBB toolchain, and the conversion only needs to read primvars (positions, scales, transfer, masks, occlusion indices) from an already-baked `ParticleField`/`Points`, not to compose a stage.

**`.athc` version 3, sections and tiers** — A custom binary container. *Section* = an array of one data type (identified by a FourCC, four ASCII characters such as `CORE` or `TXDI`); *tier* = a read quality level (1: geometry and colour; 2: material; 3: full relight; 4: the phone's compact record). The key property is that, within each block, a tier's sections form a contiguous **prefix** of the block, so reading "block $b$ up to tier $t$" is a single byte range.

**HTTP Range** — The HTTP/1.1 header `Range: bytes=a-b` (RFC 9110, §14) asks a static server for just a piece of a file, to which it answers `206 Partial Content`. It allows streaming a huge file with no special server: the client first reads the header and the tables, then requests exactly the bytes of the pages it needs. Any CDN (R2, Cloudflare) supports it.

**Skin, skeleton, clips** — For a cloud with a skeleton (the sparrow, §5.12): the per-gaussian influences (joints and weights), the joint hierarchy (`ATSK`) and the animations (`ATCL` clips).

**Steps 2–6** are the compression chain: lossless (byte planes + gzip), lossy on the transfer (clustered PCA), lossy on visibility (vector quantisation), lossy on the scalar parameters (plane quantisation), and the reorganisation of the tree for streaming. Step 7 is what the player does at run time. Each term is developed in its section.

*References:* RFC 9110 (HTTP Semantics), §14 "Range Requests"; [43] (OpenUSD); the `openusd` crate on crates.io.
:::

**What each parameter becomes.**

| parameter | master USDC | web format | measured loss | § |
|---|---|---|---|---|
| position | 3 floats | $3\times$u16 in the page's box | under $10^{-6}$ of image error (ablation) | 7.4 |
| scale, opacity | 3 + 1 floats | u8 logarithmic scale; u8 opacity in two ranges | under $10^{-6}$ (ablation) | 7.4 |
| orientation | quaternion | one 32-bit word: octahedral axis 10+10 bits, angle 12 | | 7.4 |
| albedo | float rgb | u8 sRGB | under $10^{-6}$ (ablation) | 7.4 |
| normal | float | octahedral, 12 bits an axis | relMSE $2.7\times10^{-4}$ (8+8 bits: $5.2\times10^{-2}$) | 7.4 |
| curvature | 3 floats | 3 × s8, signed logarithm in the page's range | under $10^{-6}$ (ablation) | 7.4 |
| material | packed words per Gaussian | u16 index into an exact table of the asset's materials | none (exact) | 7.4 |
| transfer | 112 f16 values | cluster id + 24 int4 weights on a shared basis | transfer relMSE 0.0054 over the car (v8.2b); image within relMSE $10^{-3}$ of the uncompressed transfer | 7.2, 7.12 |
| open directions | 256 bits | u16 index into a 4096-entry dictionary | 62.9–65.4 dB | 7.3 |
| rig | influences, gradients, zonal lobes | `SKIN` section: joint u16, weight unorm16, gradients in half; merged nodes without gradient | relMSE 0.0015–0.0053 against the engine's raster | 5.12 |
| merge tree | nodes, pages, occlusion | error-band pages, traversal records of 5–11 bytes, the occlusion codebook | the same cut | 7.5 |

::: {.nota}
**Reading the table, row by row** — Each row says how a parameter is stored in the master (USDC, 32-bit floats except the transfer in f16) and in the web format, and what error the conversion introduces:

- **Position, $3\times$u16 in the page box.** Uniform 16-bit quantisation within the page's bounding box: $x = \ell + q\,(h-\ell)/65535$. The step is $(h-\ell)/65535$; for a page 1 m across, about 15 µm, far below the size of any gaussian. Hence its image error is negligible ($<10^{-6}$ relMSE in the ablation: removing that parameter's quantisation and measuring the difference).
- **Log-scale u8.** Gaussian scales vary over orders of magnitude; quantising $\ln s$ uniformly gives a constant *relative* error (each step multiplies the scale by the same factor), which is what the eye perceives.
- **Two-segment u8 opacity.** A quadratic segment for $[0,1]$ (more resolution near 0) and a linear one for Spark's LoD opacity ($>1$). Detailed in §7.4.
- **Orientation in one 32-bit word.** The quaternion is rewritten as an axis (in octahedral projection, 10+10 bits) and an angle (12 bits): a Spark format ("ext").
- **sRGB u8 albedo.** The sRGB curve (roughly $\gamma\approx2.2$) concentrates codes in the darks, where the eye discriminates more; 8 bits in sRGB are perceptually equivalent to about 12 linear bits.
- **Octahedral normal, 12 bits per axis.** The octahedral projection (Cigolle et al. 2014) maps the unit sphere to a square; 12+12 bits give an angular error of the order of $10^{-3}$ rad. With 8+8 the angular error (of the order of $10^{-2}$ rad) is doubled in the reflection direction and the clear coat's sharp reflections shift: relMSE $5.2\times10^{-2}$ against $2.7\times10^{-4}$.
- **Curvature.** Three signed values with a large dynamic range: signed logarithm in 8 bits, normalised to the page's range.
- **Material.** An exact index into a table of the asset's distinct material combinations: lossless.
- **Transfer.** 112 f16 values (224 bytes) become 1 cluster byte and 24 4-bit weights (12 bytes): 17× less. The *transfer* relMSE (0.0054, the error on the coefficients) is not the *image* one (within $10^{-3}$ of the uncompressed transfer), because colour is a dot product with the light that averages errors out.
- **Open directions.** 256 bits (32 bytes) become a 16-bit index (2 bytes); the 62.9–65.4 dB figure is image PSNR against the exact masks.
- **Rig.** Joint u16, weight in unorm16 (unsigned integer normalised to $[0,1]$), weight gradients in f16.
- **Merge tree.** Reorganised, not approximated: "the same cut" means the traversal picks exactly the same nodes as with the original tree.

*References:* Cigolle, Z. H., et al. (2014), "A Survey of Efficient Representations for Independent Unit Vectors", JCGT 3(2); IEC 61966-2-1 (sRGB).
:::

The whole compact record is 40 bytes on the GPU (42.4 bytes resident with its tables, and 16 bytes per drawn splat),
against 256 bytes for the master's transfer and visibility alone on the device. The compact path's image is within
40.2–41.9 dB of the full relight of the same cloud. For the Corvette, the 87.4 GB of the close-up level's 24 trees
become 12.77 GB of web data (v8.2b, 149.0 M nodes), streamed by page so that a view loads a fraction of it.

::: {.nota}
**40 bytes against 256** — The master's 256 bytes on the device are the transfer (112 × 2 = 224 B) plus the open directions (32 B); the compact record's 40 bytes already include *all* the parameters. The 42.4 B "resident with its tables" spread across the gaussians the cost of the CPCA basis, the shadow dictionary and the material table. The 16 B per drawn splat are the record the shade writes for the draw pass (§7.7), and they exist only for that frame's cut.

**40.2–41.9 dB** — PSNR of the compact path's image against the full (uncompressed) relight of the same cloud: a relative mean squared error of about $10^{-4}$, invisible to the naked eye except in specular highlights.

**87.4 GB → 12.77 GB** — A factor of 6.8. The 12.77 GB are the whole published web set (83 files with all their tiers, §7.12.3); on average about 86 bytes per node for 149.0M nodes, but no phone reads more than a fraction: it does not touch tiers 1–3 and from tier 4 it requests only the pages its view needs. What matters to the phone is the download per view (tens of MB, §7.12.2), not the total.

*References:* §7.4, §7.11 and §7.12.3 of the paper.
:::

**Why the two are kept apart.** Each step of this process discards information the bake produced: a basis fitted to
one asset, a dictionary that replaces each mask by its nearest, quantisation chosen for what the eye sees at a phone's
resolution. A web data set is therefore never a master: it cannot be relit beyond what its encodings keep, re-baked,
converted to another format or compared against the path trace without the master it came from. Every improvement to
the asset is made on the master and the web format is derived again.

::: {.nota}
**Generation loss** — Each step in the chain (fitted basis, dictionary, quantisation) is a projection onto a smaller space chosen *for one use*: a CPCA basis fitted to the Corvette represents another asset poorly (§7.2.2: parts excluded from the fit up to 400 times worse); a mask dictionary does not keep rare masks; a quantisation chosen for a phone's resolution may not be enough for 4K. Recompressing already-compressed data accumulates errors without control (what video calls *generation loss*). That is why the rule is structural: the web format is a leaf of the data graph, never an input. Nor does it allow validation against the path trace on its own, because one cannot tell which part of the error comes from compression and which from the bake.

*References:* §7.2.2 (table "Basis evolution"); [15] (CPCA is a data-dependent approximation).
:::

# 7. The web format: compression, streaming and a WebGPU player

This section describes the web format of Part II and the player that draws it: the whole Corvette, with its full light transfer, drawn on an iPhone in a web page and judged against the mesh path trace, within about 50–60 MB of car data while moving and about 100 MB at rest, with small parts that look right. The player is a fork of Spark with a WebGPU backend (sparkwebGPU, with demos at sparkwebgpu.lucab.co.uk); its input is the master of Part I -- the clouds, the merge trees, the pages and the occlusion that athenea writes. This section describes the formats, the streaming, the draw and the relight on the phone, and the measured results.

::: {.nota}
**Spark** — A Gaussian splatting renderer for the web by World Labs [11], built on three.js (the JavaScript 3D library on top of WebGL): it loads PLY/SPZ/SOG, sorts and draws splats, and handles LoD and editing. Its original backend is WebGL2.

**WebGPU** — The modern graphics API of the web (W3C, [46]), successor to WebGL. It exposes the Vulkan/Metal/D3D12 model: explicit device and queue, precompiled pipelines, *compute shaders*, storage buffers and bind groups. Chrome has shipped it since 2023, Safari since version 26 (2025) on macOS and iOS. What is decisive for this player is *compute shaders*: without them there is no GPU sort, no per-gaussian relight and no GPU cull. Its shading language is **WGSL** (WebGPU Shading Language); the player writes its kernels in Slang [44] and compiles them to WGSL.

**sparkwebGPU** — The fork of Spark with a WebGPU backend, plus some athenea adapters written in Slang (the relight, the compact path) and some Rust tools for the format. athenea's drawing on the phone is a path of its own (`CompactRelitPath`), not Spark's renderer.

**Judged against the mesh path trace** — The ultimate reference is the path-traced render of the original model (triangles and MaterialX materials), not the cloud: the goal is for the iPhone to look like the production render.

*References:* [11] (Spark), [46] (W3C WebGPU: https://www.w3.org/TR/webgpu/ ; WGSL: https://www.w3.org/TR/WGSL/), [44] (Slang); athenea `docs/decisions.md` (section "A note on WebGPU": what would and would not port from the native engine).
:::

**Three kinds of measurement** appear, and they are not interchangeable:

- **Dawn / M5 Pro**: GPU timestamps from headless Dawn (Metal) on an Apple M5 Pro, at the phone's CSS resolution (393 × 852) or at 1920 × 1080. They are a proxy for the phone's GPU.
- **Emulated iPhone**: the live site in desktop Chrome on the same M5 Pro, with the page in its mobile mode at 393 × 852. The pool sizes, the cut and the memory are the phone's; the speed is not. The emulated frame rates are about twice the real iPhone's, so draw counts are judged on the device.
- **Real iPhone**: iOS 18.7, Safari 26.6.1, a 393 × 852 @3 screen (the iPhone 14 Pro–16 class), canvas 393 × 695 at pixel ratio 1. These are stress-page ramps and two whole-car runs on the device.

::: {.nota}
**Dawn** — Google's C++ implementation of WebGPU (the one Chrome uses). "Headless": run from Node through the `webgpu` package (Dawn bindings for Node), with no browser or window, on top of Metal on the Mac. *GPU timestamps* are WebGPU's optional `timestamp-query` feature: the pass writes a GPU counter at its start and end (`timestampWrites`), which measures pure GPU time, without CPU or synchronisation noise.

**M5 Pro as a stand-in** — The M5 Pro's GPU is much faster than an iPhone's, but it has the same architecture (Apple GPU, tile-based deferred rendering) and the same target shading language (Metal), so the relative changes of a kernel (−23 %, ×3) carry over better than absolute times.

**Resolutions** — 393 × 852 are *CSS points* of the iPhone 14 Pro–16 class; at pixel ratio @3 the physical screen has 1179 × 2556 pixels. The player draws at pixel ratio 1 on a 393 × 695 canvas (the height left by Safari's interface): 9 times fewer pixels than the native resolution, a deliberate choice to keep the per-pixel cost low.

**Emulated iPhone** — Chrome DevTools' device mode: the phone's viewport, pixel ratio and user agent, and the page switches on its phone configuration (pool, budget, $\tau$). The GPU is still the Mac's, which is why emulated fps do not count as a device measurement.

**iOS 18.7 and Safari 26.6.1** — This combination is what the browser reports. Since iOS 26, Safari freezes the iOS version in its user agent at 18.x, so the device was probably running iOS 26 even though the string says 18.7; the paper gives the string as read.

*References:* Dawn: https://dawn.googlesource.com/dawn ; WebGPU specification, §"Timestamp Query" ([46]).
:::

Image quality is given as PSNR in dB, or as relMSE ($\sum(x-y)^2/\sum y^2$; where PSNR is written $-10\log_{10}\mathrm{relMSE}$ it is stated). The reference is either the path trace at athenea's `/World/Camera` (1920 × 1080) or the player's own uncompressed path. The reference used is named in each case.

::: {.nota}
**PSNR and relMSE** — Two ways of measuring the difference between an image $x$ and a reference $y$:

$$
\mathrm{relMSE} = \frac{\sum_p (x_p-y_p)^2}{\sum_p y_p^2},\qquad
\mathrm{PSNR} = 10\log_{10}\frac{\mathrm{MAX}^2}{\mathrm{MSE}} .
$$

relMSE normalises by the reference's energy, so it depends neither on exposure nor on a maximum value (useful in linear HDR). When "PSNR" is written as $-10\log_{10}\mathrm{relMSE}$ it is the same logarithmic scale with the reference's energy in place of $\mathrm{MAX}^2$: relMSE $10^{-3}$ corresponds to 30 dB, $10^{-4}$ to 40 dB, $10^{-6}$ to 60 dB. Every 10 dB is a factor of 10 in squared error; 3 dB, a factor of 2. As a rule of thumb, above 40 dB a difference cannot be seen without zooming; above 60 dB it is rounding noise; figures such as 188 dB (§7.12.3) indicate identical images except for last-bit differences in some floating-point operation.

**The two references** — The path trace (athenea's `/World/Camera`, 1920 × 1080) measures *absolute* quality (all the bake error plus the compression error); "the player's own uncompressed path" measures only the error the web format adds. A figure of 19 dB against the path trace and one of 41 dB against the uncompressed relight do not contradict each other: they measure different things.

*References:* athenea `docs/decisions.md` and `docs/operations.md` (`athenea compare`: image comparison on the GPU); §2.15 and §8 of the paper. PSNR slightly favours images somewhat darker than the reference (less energy, less absolute error in the highlights), which is why the paper also gives regions and brightness ratios.
:::

## 7.1 The container: `.athc` version 3

athenea's `.athc` v2 is a flat array of words per splat. The web player rewrites it as **version 3**, which is laid out by sections and tiers. Inside each block, the data a tier needs is a prefix, so each (block, tier) pair is one HTTP Range request:

::: {.nota}
**`.athc` v2 versus v3** — athenea's v2 is an "array of structures": for each splat, all its words in a row. Good for uploading to the GPU in one go, bad for the web: one cannot read *only* the geometry without the transfer, nor request just a useful piece. v3 is a "structure of arrays" per block (a block is a LoD level or a chunk/page of splats), with the sections ordered by tier, so that the block's prefix up to tier $t$ contains everything that tier needs. The v2 → v3 → v2 conversion is byte-for-byte identical: v3 reorders, it does not change values.
:::

| Tier | Sections | Contents |
|---|---|---|
| 1 | `CORE`, `SHRS`, and optionally `LODS`, `LODE`, `OCCL`, `LODM`, `TREE`, `SKIN` | splat geometry and colour; LoD metadata |
| 2 | `MATL` | normals, emission, pbr, lobes |
| 3 | `SHAD`, `CURV`, `TXDI`, `TXIN`, `TXFD` | open-direction bits, curvature, transfer (direct, indirect, reflected field) |
| 4 | `CMPC` | the compact record for the phone (§7.4) |

::: {.nota}
**The sections, one by one** —

- `CORE` (36 B/splat): position, shape (opacity, scale, quaternion and DC colour packed together) and a tail of data; the minimum needed to draw a 3DGS gaussian [1].
- `SHRS`: the remaining colour spherical harmonics (degrees 1–3) of a classic 3DGS.
- `LODS`: a merge's error-measured LoD size (one float). `LODE` (8 B): the error-cut bounds, $\varepsilon$ and $\rho$ in f16, plus the normal cone ($\theta$ f16 and an 8+8-bit axis). `LODM`: a second opacity for merges. `OCCL`: the index into the occlusion codebook (entry + 1; 0 for no mask). `TREE` (8 B): for a paged tree, the children's page, the first child's position and their count.
- `SKIN`: influences and weight gradients of a skinned cloud.
- `MATL` (tier 2): normals, emission, pbr words and lobes (the per-gaussian material).
- `SHAD` (tier 3): the open-direction bits (256 bits per gaussian). `CURV`: the curvature (athenea's shape operator, three f16). `TXDI`/`TXIN`/`TXFD`: the transfer split into its direct half, indirect half and reflected field, so that a prefix can be read (16, 64 or 112 values).
- `CMPC` (tier 4): the phone's compact record, self-contained: a phone reads not a single byte of tiers 1–3.

**Why split into tiers** — A "3DGS only" desktop viewer reads tier 1; a viewer with materials tier 2; the full desktop relight tier 3; a phone tier 4. Each reader pays only for its own part, and there is a single file.

*References:* §5.12 (skin), §6 (LODE, OCCL, TREE) of the paper.
:::

A browser offers gzip and nothing better (`DecompressionStream`), so the lossless lever is what gzip is given. **Encoding 1** transposes each array into byte planes: byte 0 of every element, then byte 1, and so on. The high byte of an f16 or f32 changes little from one splat to the next in Morton order, so it forms runs that gzip finds. **Encoding 2** does the same after replacing each 16-bit lane with its wrapped difference from the previous element. `athc-convert --gzip --planes` picks the smaller encoding per section. Measured on the published clouds: Corvette light 68.1 → 60.9 MB (−10.6 %), Corvette HD 288.7 → 248.6 MB (−13.9 %), its paint −23 %, pawn parts −17 to −33 %. The shadow bits are random, and they never gain.

::: {.nota}
**gzip, DEFLATE and `DecompressionStream`** — gzip (RFC 1952) is a wrapper around DEFLATE (RFC 1951): LZ77 (replacing repetitions with back-references within a 32 KB window) followed by Huffman codes. The `DecompressionStream` API (WHATWG Compression Streams standard) decompresses `"gzip"`, `"deflate"` and `"deflate-raw"` natively and as a stream; it does not offer zstd, and brotli is not in every engine. A WASM decompressor (zstd, for instance) would be possible, but slower and heavier than the native one. That is why the lever is **preparing the data** so that DEFLATE compresses it better, not changing the compressor.

**Encoding 1: byte planes** — Transposition: for $n$ elements of $w$ bytes, first the $n$ bytes 0 are written, then the $n$ bytes 1, and so on (the same filter as Blosc's or HDF5's *shuffle*). The intuition with an f16 (1 sign bit, 5 exponent bits, 10 mantissa bits): its high byte holds the sign, the exponent and the top 2 mantissa bits, and changes little between neighbouring splats; the low byte is almost noise. Interleaved, the noise breaks every repetition; separated, the high byte forms long runs that LZ77 finds and Huffman codes in few bits. **Morton order** (Z-curve: interleaving the bits of the x, y, z coordinates) makes consecutive splats in the file neighbours in space, which makes neighbouring values similar.

**Encoding 2: delta planes** — Before transposing, each 16-bit lane is replaced by $v_i - v_{i-1} \bmod 2^{16}$ (with wrap-around, i.e. modular arithmetic, exactly reversible). It works better when values vary smoothly (the differences are small and their high bytes almost always 0 or 0xFF).

**The figures** — Choosing the best encoding per section saves 10–14 % over plain gzip on the Corvette, more on the paint (−23 %, a large smooth surface) and on the pawn (−17 to −33 %). The shadow bits are, to a compressor, nearly random (each bit depends on distant geometry, not on the position in Morton order), with an entropy close to 1 bit per bit: no reordering helps. Vector quantisation will solve that (§7.3).

*References:* RFC 1951 (DEFLATE), RFC 1952 (gzip); WHATWG Compression Streams: https://compression.spec.whatwg.org/ ; Morton, G. M. (1966), IBM technical report.
:::

## 7.2 Transfer compression by clustered PCA

The transfer is 30–70 % of a relit cloud's bytes, and neighbouring transfers are close. Two clustered-PCA codings are used: a lossy encoding of the file (encoding 3), and a joint basis that the phone keeps resident.

::: {.nota}
**The transfer** — It is the core of the relight (§4): for each gaussian, a vector of coefficients that, multiplied by the light expressed in spherical harmonics (SH), gives its outgoing radiance. It is Sloan et al.'s Precomputed Radiance Transfer idea [14]: the lighting integral $\int L(\omega)\,T(\omega)\,d\omega$ becomes, with $L$ and $T$ projected onto an orthonormal basis, a dot product $\sum_k l_k t_k$. The 112 values are 16 direct (degree-3 SH, scalar: sky visibility with the cosine), 48 indirect (16 × RGB: light bounced off the object itself) and 48 for the reflected field (16 × RGB: what is seen reflected in each direction, for the specular).

**Why neighbours look alike** — The transfer is a smooth function of position and normal: two adjacent gaussians on the same surface see almost the same sky and the same surroundings. The transfer vectors of a block therefore live close to a low-dimensional subspace, which is exactly what a PCA exploits.

*References:* [14] (PRT), [15] (CPCA), [17] (irradiance with 9 SH); §4 of the paper; athenea `docs/decisions.md` (section "A transfer right under any sky and any light (task TX)").
:::

### 7.2.1 Encoding 3 (file level)

Following Sloan et al. (2003), a transfer section of one block is stored as up to $K$ clusters (k-means on a sample; $K \le n/128$). Each cluster $c$ has a mean $\mu_c$ and $m_c \le M$ principal directions $\mathbf{d}_{c,j}$ in f16. Each element $i$ has a cluster id and $m_c$ quantised coefficients:

::: {.nota}
**k-means** — Partitioning $n$ vectors into $K$ groups by minimising the sum of squared distances to each group's mean; Lloyd's algorithm [23] alternates assigning each point to the nearest centre and recomputing each centre as the mean of its points. Here it is fitted on a uniform sample of the block and then each element is assigned to its nearest centre. The bound $K \le n/128$ guarantees at least about 128 elements per cluster on average, so that the cluster's mean and principal directions come from enough data and their cost (stored in f16) is amortised.

**Per-cluster PCA** — Principal component analysis: the directions $\mathbf d_{c,j}$ are the eigenvectors of the cluster's covariance matrix, $\mathrm{Cov}_c = \frac{1}{n_c}\sum_{i\in c}(\mathbf t_i-\mu_c)(\mathbf t_i-\mu_c)^\top$, sorted by decreasing eigenvalue. Truncating to the first $m_c$ is the best rank-$m_c$ approximation in squared error (Eckart–Young theorem). Combining the two ideas is the *Clustered PCA* (CPCA) of Sloan et al. [15], designed precisely for PRT: a cluster captures the region, the local PCA captures its variation.

*References:* [15], [23]; Jolliffe, I. T. (2002), *Principal Component Analysis*; Eckart, C., Young, G. (1936), Psychometrika.
:::

$$
\hat{\mathbf{t}}_i = \mu_{c(i)} + \sum_{j<m_{c(i)}} \left(q_{ij}\,\Delta_{c(i)}\right)\mathbf{d}_{c(i),j},
\qquad q_{ij}\in\mathbb{Z},
$$

::: {.nota}
**The reconstruction** — Each element is reconstructed as its cluster's mean plus a linear combination of the cluster's directions. The actual coefficient is $q_{ij}\,\Delta_{c}$: an integer times a uniform quantisation step for the cluster. If $\mathbf t_i$ is a vector of dimension $D$, $m_c$ small integers are stored instead of $D$ halves, and the mean and directions are paid for once per cluster.

*References:* [15] (CPCA reconstruction equation).
:::

where every product and sum is rounded to f32 in that order and the result is rounded to the nearest half. This makes the Rust/WASM and TypeScript decoders bit-identical. The step $\Delta_c$ is proportional to the cluster's RMS, which gives the same relative error in dim clusters as in bright ones. The coefficients are zig-zag integers of 1–3 bytes, stored as byte planes. $m_c$ is the smallest count whose truncation takes at most 40 % of the error budget. The budget is relative per block:

::: {.nota}
**Bit-identical decoding** — Floating-point addition is not associative: $(a+b)+c \ne a+(b+c)$ in general. If the Rust decoder (compiled to WASM, for whole files and pages) and the TypeScript one (for loose streams of one chunk) summed in a different order or at a different precision (JavaScript uses doubles by default), they would give different halves and the image would depend on which path loaded each block. By fixing the recipe — every product and every sum rounded to f32 (in JS with `Math.fround`) in a given order, and a final rounding to the nearest half with ties to even — both produce exactly the same bits, and a test checks it.

**Step proportional to the RMS** — With $\Delta_c \propto \mathrm{RMS}_c$, the quantisation error (uniform, variance $\Delta^2/12$ per coefficient) is proportional to the cluster's typical magnitude: the *relative* error is the same in dark clusters (small transfer, shadowed areas) as in bright ones. Without this, a global step would wreck the dark areas, where the eye is more sensitive to relative errors.

**Zig-zag** — Maps signed integers to unsigned ones by interleaving: $0\to0, -1\to1, 1\to2, -2\to3,\dots$; in bits, $z = (q \ll 1) \oplus (q \gg 31)$. A small integer in absolute value thus has zero high bytes, and in byte planes those high planes compress to almost nothing. It is Protocol Buffers' encoding.

**Splitting the budget: 40 % truncation, 60 % quantisation** — The total error has two sources: the discarded directions (truncation) and the rounding of the coefficients. The smallest $m_c$ whose truncation consumes at most 40 % of the budget $E$ is chosen, and the step $\Delta$ is adjusted so that the total fits within $E$.

*References:* IEEE 754-2019 (round to nearest, ties to even); Protocol Buffers, "Encoding" (ZigZag): https://protobuf.dev/programming-guides/encoding/ .
:::

$$
E_{\text{block}} = \frac{\sum_i \lVert \hat{\mathbf t}_i - \mathbf t_i\rVert^2}{\sum_i \lVert \mathbf t_i\rVert^2} \le E, \qquad E = 10^{-5}\ \text{by default},
$$

::: {.nota}
**The relative error budget** — $E_{\text{block}}$ is a relMSE over the block's coefficients. With $E = 10^{-5}$ (the default) the reconstructed transfer is about 50 dB from the original: a very small error, meant to make encoding 3 "nearly lossless" in the desktop file.
:::

and the step shrinks until the budget holds. Because the SH transfer coefficients are orthonormal, $E$ is the $L^2$ error over the sphere. A block that cannot meet $E$, or that byte planes store smaller, keeps its values exactly. Measured: the transfer sections take 4–7× fewer bytes (the pawn top's TX 112 went from 157 to 28 MB); a relit splat moves by relMSE $4\times10^{-8}$ to $3\times10^{-5}$; the Corvette light set went from 67.9 to 49.5 MB and the pawn page from 281 to 84 MB.

::: {.nota}
**Why $E$ is the $L^2$ error over the sphere** — The spherical harmonics $Y_k$ are orthonormal: $\int_{S^2} Y_k Y_l\,d\omega = \delta_{kl}$. For a function $f=\sum_k t_k Y_k$, Parseval's identity gives $\int_{S^2}(\hat f - f)^2\,d\omega = \sum_k(\hat t_k - t_k)^2$. The error measured on the coefficients is exactly the error of the function over the sphere, which is what the product with the sky, the evaluation of the sun at a point and the reflection of the field read.

**A block that does not meet it** keeps its exact values (mode 0, byte planes only): the encoding never makes a block worse than the budget.

**The figures** — 4–7× fewer bytes in the transfer sections (157 → 28 MB: 5.6× on the pawn's cap with a 112-value TX). A relit splat moves by between $4\times10^{-8}$ and $3\times10^{-5}$ relMSE (between 75 and 45 dB): invisible. The Corvette's light set goes from 67.9 to 49.5 MB (−27 %), less than the sections' factor because the transfer is only part of the file.

*References:* [16] (properties of SH), [17].
:::

### 7.2.2 The joint basis kept on the GPU

For the phone, the transfer is never expanded. athenea's joint CPCA splits the 112 TX values into three blocks $b \in \{\text{direct } (16), \text{indirect } (48), \text{reflected } (48)\}$. It uses $C$ clusters shared by the three blocks and $K = 8$ directions a block. A gaussian stores one cluster id $c$ and $3 \times 8$ weights:

::: {.nota}
**"The transfer is never expanded"** — On the desktop the transfer is decompressed to 112 halves per gaussian on the GPU. On the phone that would cost 224 B per gaussian (224 MB per million), which is unacceptable; the shade works directly with the weights and the basis.

**Joint CPCA** — Unlike encoding 3 (clusters per file block), here there is **one** basis for the whole asset and the three blocks share the same cluster id: a single cluster byte per gaussian. Clustering is done on the joint 112-value vector (each block normalised by its energy so that the reflected one, of larger magnitude, does not dominate), followed by a PCA per cluster and per block. $K=8$ directions per block is a fixed compromise: the record needs a constant size (12 bytes of weights) so that the GPU can address it by index.

*References:* [15]; athenea `shaders/athenea/lod/transfer_codec.slang` (header: athenea's basis codec, "direct 16, indirect 48 and reflected 48 coefficients, each divided by its block's energy norm").
:::

$$
\mathbf t^{(b)}_i \;\approx\; \mu^{(b)}_{c} + \sum_{k=1}^{8} q^{(b)}_{ik}\,\frac{s^{(b)}_{c,k}}{127}\,\mathbf B^{(b)}_{c,k},
$$

::: {.nota}
**The formula** — $\mathbf t^{(b)}_i$ is block $b$ of gaussian $i$'s transfer; $\mu^{(b)}_c$ the cluster's mean in that block; $\mathbf B^{(b)}_{c,k}$ axis $k$; $s^{(b)}_{c,k}$ that axis's scale (the range of weights it needs); $q^{(b)}_{ik}$ a signed 8-bit integer in $[-127,127]$. The actual weight is $q\,s/127$, which ranges from $-s$ to $s$.
:::

where $\mathbf B$ are the per-cluster axes and $s$ their scales. On the GPU, each direction is stored premultiplied by $s/127$ in f16, 504 words a cluster. **Weights** are int8 or, on the phone, int4: encoded at 7 levels and decoded as $q_8 = \mathrm{round}(q_4 \cdot 127/7)$. On the phone, int4 costs $+0.7\times10^{-4}$ of per-splat colour relMSE and $0.1$–$4.6\times10^{-5}$ of image relMSE; the CPCA's own truncation dominates.

::: {.nota}
**504 words per cluster** — Per cluster the mean (112 values) and the axes ($8\times112 = 896$ values) are stored: 1008 halves = 504 32-bit words. Premultiplying each axis by $s/127$ saves one multiplication per weight in the shade and leaves the decode as $\mu + \sum_k q_k \mathbf B'_k$. For 256 clusters, $256\times504\times4 = 516\,096$ bytes: the "516 KB" of §7.2.2.

**int8 versus int4** — On the phone each weight uses 4 signed bits, with values in $[-7, 7]$ ("7 levels" per side, 15 in total). It is decoded onto the int8 grid with $q_8 = \mathrm{round}(q_4\cdot127/7)$, so that the shader treats both widths alike. The (uniform) quantisation error has variance $\Delta^2/12$; going from 255 to 15 levels multiplies $\Delta$ by 17, but the resulting error ($+0.7\times10^{-4}$ relMSE of per-splat colour) is still smaller than that of the CPCA's own truncation to 8 directions. That is why "the truncation dominates": spending bits on the weights would not pay off.
:::

**The projected route.** Under one dome, only the reflected block depends on the eye. Each relight, a thread per cluster projects the cluster's mean and 8 axes of the direct and indirect blocks onto the light (the dome's 16 harmonics, the sun direction, the white-sky coupling): 144 words a cluster. The view-independent ("kept") terms of a gaussian are then dot products of its weights with those projections. Per frame, only the 48 reflected values are rebuilt, in registers. On real data this route reproduces the full rebuild to relMSE $3\times10^{-11}$.

::: {.nota}
**The projected route: linearity** — The conserved diffuse colour (direct and indirect blocks) is linear in the transfer: $c = \mathbf t\cdot\mathbf L$, where $\mathbf L$ are the light's SH coefficients. Substituting the reconstruction,

$$
\mathbf t^{(b)}_i\cdot\mathbf L = \mu^{(b)}_c\cdot\mathbf L + \sum_{k=1}^{8} q^{(b)}_{ik}\,\big(\mathbf B'^{(b)}_{c,k}\cdot\mathbf L\big).
$$

The products in brackets depend only on the cluster and the light, not on the gaussian: they are computed once per cluster when the light changes (one thread per cluster) and each gaussian only does 8+8 multiply-adds with its weights. This is exactly Sloan et al.'s "trick" [15]: in CPCA, the per-vertex render cost is proportional to the number of components $m$, not to the dimension of the transfer.

**144 words per cluster** — 18 vectors (mean + 8 axes, in two blocks) are projected onto the dome (16 RGB harmonics), the sun direction and the white-sky coupling: about 8 words per projected vector, 144 in total; $256\times144\times4 = 147\,456$ bytes, the "147 KB" of §7.2.2.

**The reflected block** depends on the eye (the reflection direction changes with the camera), so it cannot be projected in advance: every frame, each gaussian rebuilds its 48 values in registers and evaluates them along its mirror direction.

**$3\times10^{-11}$** — A difference of pure floating-point rounding (about 105 dB): the projected route is mathematically identical to the full reconstruction.

*References:* [15], [14].
:::

**Basis evolution.**

| basis | clusters | fitted on | relMSE of the transfer | notes |
|---|---|---|---|---|
| athenea's basis | 256 | athenea's sample of `base_tx` | 1.69e-2 (whole car) | 1.17 MB f32 file; the player's cluster assignment matches athenea's 100 % |
| player fit, six parts | 256 | six parts of the car | direct 2.7e-4, indirect 2.5e-3, reflected 1.1e-3 | in-sample; held-out pieces 10–400× worse |
| player fit, whole car | 256 | every second node of the car | 1.03e-2 (direct 1.6e-3, indirect 8.0e-3, reflected 1.06e-2) | image within relMSE 1e-3 of the uncompressed transfer |
| player fit, 512 clusters (v8 data) | 512 | car1280 + car5120 | fine3_0: 0.0181 → 0.0136 | every 1280 panel group −20 to −30 % |

::: {.nota}
**The basis table** —

- **athenea's basis (q97).** Fitted on a sample of `base_tx` (the Corvette's bake with transfer), 256 clusters, stored in f32 (1.17 MB). Transfer relMSE 1.69e-2 over the whole car. That the cluster assignment matches 100 % confirms that the player and the engine apply the same assignment rule (nearest centroid with energy normalisation).
- **Player fit on six parts.** Much better *within the sample* (direct 2.7e-4), but 10–400 times worse on unseen parts: it is *overfitting* to the distribution of those parts. Lesson: the basis must be fitted on every part it is going to encode.
- **Fit on the whole car.** With one node in two (`--sample 2`, to fit in RAM: 448 bytes per fitted node) it improves on q97 (1.03e-2 against 1.69e-2); in the image it is within relMSE $10^{-3}$ (30 dB) of the uncompressed transfer. That the reflected block is the worst one (1.06e-2) is natural: it is the one that varies most (it depends on high-frequency reflections).
- **512 clusters (v8 data).** Doubling the clusters reduces the error by 20–30 % per panel group (on `fine3_0`, from 0.0181 to 0.0136).

*References:* athenea `shaders/athenea/lod/transfer_codec.slang`.
:::

The fit (`fit-compact-basis`) normalises the energy per block, runs k-means (k-means++ seeds, Lloyd over every node) on the joint vectors, then a per-cluster PCA of each block, and chooses the scale per direction with the least error through the int4 code. The **512-cluster basis** keeps the record size: the 9th bit of the cluster id goes into the free bit 15 of the shadow-dictionary index, which caps the dictionary at 32 768 entries (the car's has 4096). Files with 256 clusters read as before. Cost: basis 516 KB → 1.03 MB (f16), projections 147 → 295 KB, tier 4 +1.6–1.8 %, and the same shade time on the M5 Pro (2.82–2.88 ms at 400k drawn). It has not been measured on the real phone.

::: {.nota}
**The fit, step by step** —

1. *Normalise the energy per block*: divide each block by its mean norm so that all three weigh the same in the k-means distance.
2. *k-means++* [23]: pick the seeds at random with probability proportional to the squared distance to the nearest seed already chosen, which gives an initial solution within a factor $O(\log K)$ of the optimum in expectation; then Lloyd over all the nodes.
3. *PCA per cluster and block*: diagonalise the covariance (Jacobi's method, which is simple and robust for symmetric matrices up to 48 × 48).
4. *Quantisation-aware per-direction scale*: instead of taking $s$ as the absolute maximum of the coefficients (which wastes levels on one outlier), the $s$ that minimises the error after going through the int4 code and the int8 grid is searched for. It is the classic balance between clipping error and granular error.

**The 9th bit** — With 512 clusters the id needs 9 bits, but the record has 1 cluster byte. The shadow index is u16 and the car's dictionary has 4096 entries (12 bits), so bit 15 is free: it is used as bit 8 of the cluster (`clusterAt` reads `cluster | (shadow >> 15) << 8`). The price is that the dictionary is limited to $2^{15} = 32\,768$ entries. The record stays at 40 bytes, and 256-cluster files (bit 15 always 0) read the same.

**Cost** — The basis doubles ($512\times504\times4 \approx 1.03$ MB), so do the projections (295 KB), tier 4 grows by 1.6–1.8 % (more entropy in the cluster byte). The shade does not change (2.82–2.88 ms with 400k drawn on the M5 Pro): the number of clusters does not alter the per-gaussian work, only the size of the tables.

*References:* [23] (Lloyd; Arthur and Vassilvitskii, k-means++); Golub, G. H., Van Loan, C. F., *Matrix Computations* (Jacobi's method).
:::

## 7.3 The open-direction bits by vector quantisation

Each gaussian's 16 × 16 open-direction cells (256 bits) are replaced by an index into a dictionary (`athc-convert --shadow-vq`). The dictionary is k-means in Hamming space, each centroid the bitwise majority of its members with ties kept open, 12 rounds, seeded from member $\lfloor c\,n/C\rfloor$. With 1024 centroids a piece, the image error is 62.9–65.4 dB (athenea's figure: 62 dB), and the mean Hamming distance is 2.0–2.5 of 256 (p99 14–16). The raw masks as a dictionary would be 15.3 MB at 1M splats, against 1.51 MB for the VQ. Across six pieces, one asset dictionary of 4096 entries (131 KB, mean distance 1.07) beats six dictionaries of 1024 (197 KB, mean distance 1.00) on bytes for about the same error, so the shared dictionary is the default.

::: {.nota}
**Open directions** — For each gaussian, the bake (§4) traces a ray through each of the $16\times16 = 256$ cells of an octahedral projection of the hemisphere/sphere of directions and stores one bit: 1 if the direction is *open* (the ray escapes to the sky), 0 if it is closed (it hits the object itself). The relight uses them so that the sun and the sharp reflections respect occlusion with more angular resolution than degree-3 SH. Each cell covers about $11^\circ$ (§7.13).

**Vector quantisation (VQ)** — Instead of storing each vector (here a 256-bit mask), an index into a dictionary (*codebook*) of $C$ representative vectors is stored, and each mask is replaced by the nearest entry. It is the classic lossy compression scheme of Linde–Buzo–Gray and Gersho–Gray; its training is a k-means in the appropriate metric.

**k-means in Hamming space** — The distance between masks is the Hamming distance: the number of differing bits, $d_H(a,b)=\mathrm{popcount}(a\oplus b)$. For this metric, the "centroid" that minimises the sum of distances to the members is not the mean but the **bitwise majority**: each bit of the centroid is 1 if at least half the members have it set (it is the per-coordinate median, the binary analogue of k-medians; also known as *k-modes* for categorical data). **Ties left open** is a conservative decision: when in doubt, the direction is marked open, which lets light through instead of inventing shadow. The seed of centroid $c$ is the mask of member $\lfloor c\,n/C\rfloor$: fixed-stride sampling in the cloud's order (a spatial order, because the cloud is in the order of a tree walk), deterministic and spread over the whole part. 12 rounds of assignment + update are enough to settle.

**The figures** — With 1024 centroids per part, the mean distance is 2.0–2.5 bits out of 256 (fewer than 1 % of the cells wrong), with a p99 of 14–16 (the worst 1 % has about 15 cells changed, 6 %). In the image, 62.9–65.4 dB: invisible. The distinct masks as they are (477 206 in the measurement) would take $477\,206\times32 \approx 15.3$ MB as a dictionary; the VQ dictionaries of 62 parts (47 229 masks), 1.51 MB: 10× less. A 4096-entry asset dictionary takes $4096\times32 = 131\,072$ B; six of 1024, $6\times1024\times32 = 196\,608$ B. The shared one wins because masks repeat across parts (a door panel and a wing panel see similar situations) and because the fixed per-part cost (32 KB) does not scale down with small parts.

*References:* Linde, Y., Buzo, A., Gray, R. M. (1980), "An Algorithm for Vector Quantizer Design", IEEE Trans. Communications; Gersho, A., Gray, R. M. (1992), *Vector Quantization and Signal Compression*; Huang, Z. (1998), "Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values" (k-modes); [23].
:::

## 7.4 The compact record

The compact path holds a gaussian as byte planes in one storage buffer. Each plane is 64-byte aligned and read by `fieldAt(plane, width, i)`:

::: {.nota}
**Storage buffer** — In WebGPU, a buffer bound to a shader with the `"storage"` type (in WGSL, `var<storage, read>` or `var<storage, read_write>`): GPU memory of arbitrary size, addressable by index, unlike a *uniform buffer* (small, constant). It is the way to give a compute shader large arrays.

**Byte planes in a single buffer** — A structure of arrays: all the position bytes of all the gaussians, then all the albedo bytes, and so on, each plane starting on a 64-byte boundary. WGSL has no 8- or 16-bit types in storage (only `u32`, `f32`, `i32`, and `f16` as an optional extension), so `fieldAt(plane, width, i)` computes the byte $\text{plane} + i\cdot\text{width}$, reads the 32-bit word or two words that contain it and extracts the `width` bytes with shifts and masks. A single buffer avoids spending *bindings*: WebGPU limits the storage buffers per stage (`maxStorageBuffersPerShaderStage`, 8 by default on basic adapters and on many phones).

*References:* [46] (WebGPU, §"Buffer Usages" and §"Limits"; WGSL, §"Address Spaces").
:::

| plane | bytes | encoding |
|---|---:|---|
| position | 6 | $3\times$u16 in the page's box: $x = \ell + q\,(h-\ell)/65535$ |
| albedo | 3 | u8 sRGB |
| opacity | 1 | $q\le191$: $\alpha=(q/191)^2$; $q\ge192$: $\alpha = 1 + (q-192)/63$ (Spark's LoD opacity, $>1$) |
| scale | 3 | $s = \exp(\ell + (q-1)\,\delta)$, $\delta = (h-\ell)/254$; $q=0$ means exactly 0 (surfels) |
| rotation | 4 | quaternion word: octahedral axis 10+10 bits, angle 12 |
| normal | 3 | octahedral, 12 bits an axis |
| curvature | 3 | $s8$: $\kappa = \pm f\,(2^{|q|R/127}-1)$, signed log in the page's range |
| material | 2 | u16 into an exact table (pbr + lobes, 4 words) |
| cluster | 1 | u8 (9th bit in the shadow index) |
| weights | 12 | $3\times8$ int4 |
| shadow | 2 | u16 into the dictionary |
| **GPU total** | **40** | |

::: {.nota}
**The table's formulas** —

- **Position**: $x = \ell + q\,(h-\ell)/65535$, with $[\ell,h]$ the page box (in its header) and $q\in[0,65535]$.
- **Albedo**: one byte per channel in sRGB; the shader converts it to linear.
- **Opacity** in two segments: for $q\le191$, $\alpha = (q/191)^2$, a quadratic curve that gives fine steps near 0 ($d\alpha/dq = 2\sqrt\alpha/191$: at $\alpha=0.01$ the step is $\approx 0.001$, 10 %). With a linear 8-bit code the step would be $1/255 \approx 0.0039$ and $\alpha=0.01$ would land on code 3 (0.0118): an 18 % error, and the intermediate values of the glass sheets would disappear. For $q\ge192$, $\alpha = 1 + (q-192)/63 \in [1,2]$: Spark's *LoD opacity*, a value greater than 1 that marks a merged node as a "widened splat" (§7.7).
- **Scale**: $s = \exp(\ell + (q-1)\delta)$, $\delta = (h-\ell)/254$ with $[\ell,h]$ the range of $\ln s$ in the page; $q=0$ is exactly 0, which marks a degenerate axis: a *surfel* (a flat gaussian, 2DGS [5]; surfels [6]).
- **Rotation**: the quaternion as an axis (octahedral 10+10 bits) and an angle (12 bits).
- **Normal**: octahedral 12+12 bits = 3 bytes.
- **Curvature**: $\kappa = \pm f\,(2^{|q|R/127}-1)$ with $q$ an s8: a signed logarithmic code; $R$ and $f$ set the page's range. Small values with good relative resolution, large ones without overflow.
- **Material**: u16 into a table of 4 words per material (pbr + lobes) for the asset.
- **Cluster**: u8 (+ the shadow's bit 15 with 512 clusters).
- **Weights**: $3\times8$ int4 = 96 bits = 12 bytes.
- **Shadow**: u16 into the open-direction dictionary.

Sum: $6+3+1+3+4+3+3+2+1+12+2 = 40$ bytes.

*References:* [5], [6], [11] (Spark's LoD opacity).
:::

The planes are what the data forced. An 8+8-bit normal shifted the coat's reflections by relMSE $5.2\times10^{-2}$; at 12 bits the error is $2.7\times10^{-4}$. A linear opacity code destroyed the glass sheets' 0.01 alphas. The car has 17–22k distinct materials, so a u8 index could not hold them. In an ablation, position, opacity, albedo, scale and curvature each contribute under $10^{-6}$ of image error. Against the full relight on the same cloud, the compact path's per-splat colour is within relMSE $6.3$–$7.3\times10^{-4}$. The image is within 40.2–41.9 dB at 0.4–1.0M splats.

::: {.nota}
**"The planes are what the data forced"** — Each bit width was set by measuring, not by convention:

- **12-bit normal.** A specular reflection follows the mirror direction $\mathbf r = 2(\mathbf n\cdot\mathbf v)\mathbf n - \mathbf v$, which turns twice as much as the normal. A clear coat has very low roughness (lobes of a few degrees), so a normal error of a few tenths of a degree moves the reflection visibly. 8+8 bits: relMSE $5.2\times10^{-2}$ (13 dB in that region); 12+12: $2.7\times10^{-4}$ (36 dB).
- **Non-linear opacity**: the glass sheets have $\alpha\approx0.01$ (§7.4, table).
- **u16 material**: 17–22k distinct materials (textures baked per gaussian give unique pbr/lobe combinations): they do not fit in 256.
- **Ablation**: undoing the quantisation of one parameter at a time and measuring the change in the image; position, opacity, albedo, scale and curvature stay below $10^{-6}$ (60 dB), so the record's total error comes from the transfer, the normal and the shadows.

**$6.3$–$7.3\times10^{-4}$ per splat, 40.2–41.9 dB in the image** — The per-splat error (each gaussian's colour against the full relight) is larger than the image error because in the image each pixel blends many gaussians and their uncorrelated errors average out.
:::

Against the earlier relight's 308 B a gaussian, this path keeps **42.4 B** resident (planes plus tables) and **16 B per drawn splat** (the sort buffers reused as draw records). The fixed cost drops from 29 MiB to about 3 MiB. At 1M splats that is 58.6 MiB, against 326 MiB before (Dawn, buffer sizes).

::: {.nota}
**The memory count** — With $N$ resident gaussians and $D$ drawn:

$$
M_{\text{compact}} \approx 42.4\,N + 16\,D + 3\ \text{MiB},\qquad M_{\text{earlier}} \approx 308\,N + 29\ \text{MiB}.
$$

With $N = D = 10^6$: $42.4\ \text{MB} = 40.4$ MiB, $16\ \text{MB} = 15.3$ MiB, plus 3: 58.7 MiB (the paper: 58.6). Before: $308\ \text{MB} = 293.7$ MiB + 29 ≈ 323 MiB (326 measured). A factor of 5.6.

**"The sort buffers reused"** — For each drawn splat the sort needs a key and a value (two words) and their ping-pong buffers; once sorted, the shade writes the draw record (16 B) into those same buffers, because the sort no longer needs them. It is *aliasing* of temporary memory: the cost per drawn splat does not grow by having draw records.

**The fixed cost** (29 → 3 MiB) was buffers sized to the maximum (relit records, pixel records, accumulator) that the compact path does not have (§7.7).
:::

## 7.5 Error-band pages and the compact tier

### 7.5.1 The error cut

Each merged node $n$ carries the bounds $\varepsilon_n$ (error) and $\rho_n$ (sphere radius) in f16 (`LODE`). With $p$ the pixels per unit at unit distance, the traversal refines $n$ when

::: {.nota}
**The error cut and its bounds** — Each merge $n$ replaces its subtree; $\varepsilon_n$ is a bound (in world units) on how much the merge differs from the subtree it represents, and $\rho_n$ the radius of a sphere centred on the node that contains the whole subtree. Both are stored in f16 in the `LODE` section (rounded up so that they remain bounds). $p$ is the number of pixels one world unit covers at distance 1: for a perspective camera $H$ pixels tall with vertical field $\phi$, $p = H/(2\tan(\phi/2))$.

*References:* §6 of the paper; [35] (Nanite: projected error as the cut criterion, and paging by error); [2] (gaussian hierarchies).
:::

$$
P_n = \frac{\varepsilon_n\, p}{\max(d_n - \rho_n,\ \text{near})} > \tau .
$$

::: {.nota}
**The refinement test** — $P_n$ is the node's error projected in pixels: the error in world units times the pixels per unit at that distance. The denominator uses $d_n-\rho_n$, the distance to the nearest point of the sphere, not to the centre: it is the worst possible distance for any part of the subtree, which makes the test conservative. `near` avoids division by zero or by a negative number when the eye is inside the sphere (then it always refines). If $P_n>\tau$, the node is replaced by its children; otherwise it is drawn. The *cut* is the set of drawn nodes: a frontier that cuts across the tree.

*References:* [35]; athenea `shaders/athenea/lod/lod_tree_cut.slang`.
:::

$\tau$ is in pixels. athenea pages its merge trees by **error bands**, as Nanite does. A node is needed exactly when its parent's projected error exceeds $\tau$, so each node goes into the band (an octave) of its parent's error. The bands are ordered coarse to fine, each in leaf order, and the order is cut into pages of $P = 3000$ nodes. A page-table row holds a box around the parents' spheres and the largest and smallest parent error. A page is needed when

::: {.nota}
**$\tau$ in pixels** — The threshold of visible error: with $\tau = 1$ no drawn merge can differ from its subtree by more than one pixel on screen; with $\tau=3$, three. It is the main quality-versus-cost control: the number of drawn nodes scales roughly as $1/\tau^2$ (§7.10).

**Error bands, as in Nanite** — A node is in the cut only if its **parent** was refined, i.e. if $P_{\text{parent}}>\tau$. The question "do I need this node?" is therefore answered by the parent's error. Grouping the nodes by the octave ($\lfloor\log_2\varepsilon_{\text{parent}}\rfloor$) of their parent's error, a whole band stops being needed as soon as $\tau$ and distance push its maximum projected error below $\tau$. Bands are ordered from coarse (large error, needed from far away) to fine; within each band, in leaf (spatial) order, and the sequence is cut into pages of $P = 3000$ nodes. A page thus groups nodes that are needed together: the same error range and the same area.

**The page-table row** — A box that contains the spheres of its nodes' *parents*, and the maximum and minimum parent error. With that the player decides whether it needs a page without reading its nodes.

*References:* [35] (Karis et al., "Nanite: A Deep Dive", the cluster hierarchy with monotonic error and its streaming); athenea `shaders/athenea/lod/lod_pages.slang`.
:::

$$
\frac{\varepsilon^{\max}_{\text{page}}\, p}{\max(\operatorname{dist}(\text{eye}, \text{box}),\ \text{near})} > \tau \quad(\text{page 0 always}).
$$

::: {.nota}
**The page test** — It is the node test applied to the page's worst case: the largest parent error $\varepsilon^{\max}_{\text{page}}$ and the smallest possible distance, that from the eye to the box. Page 0 (the roots) is always loaded: that is what allows something to be drawn from the very first moment.
:::

Because the box bounds the parents' spheres, this test is conservative against the traversal. On a set of car parts, the settled paged cut was identical to the fully resident cut; residency was 2.27–2.53× the cut.

::: {.nota}
**Conservative** — If a node $n$ is needed, its parent $p$ satisfies $\varepsilon_p\,\mathrm{px}/(d_p-\rho_p) > \tau$. The page box contains $p$'s sphere, so the distance from the eye to the box is $\le d_p-\rho_p$, and $\varepsilon^{\max}_{\text{page}}\ge\varepsilon_p$: the page test gives a value greater than or equal and also exceeds $\tau$. No needed node is left in a page that was not requested. The price is over-loading: the resident set was 2.27–2.53 times the cut (nodes of needed pages that do not end up in the cut: ancestors, nodes pruned by the culls and nodes below the cut). This overhead will motivate v4 (§7.5.2).
:::

Two culls are applied in the traversal:

- **Back faces.** A node of a closed part is pruned with its subtree when $\mathbf n\cdot\mathbf v < -\sin(\theta + \arcsin(\rho/d))$, with $\theta$ the half-angle of its normal cone.
- **Occlusion.** With the eye outside the asset's box, a node is pruned when the four bilinear cells toward the eye ($\mathrm{oct}(\mathbf v)\cdot16-0.5$) are all closed in its occlusion-codebook entry.

::: {.nota}
**Back-face cull by normal cone** — Each node stores a cone that contains the normals of its whole subtree: an axis $\mathbf n$ and a half-angle $\theta$. Seen from the eye, the node's sphere subtends a half-angle $\arcsin(\rho/d)$, so the view directions towards any point of the node deviate from the direction to the centre $\mathbf v$ by at most that angle. If even in the most favourable case (normal turned by $\theta$ and view turned by $\arcsin(\rho/d)$) all the normals still face away, the whole subtree is invisible: $\mathbf n\cdot\mathbf v < -\sin(\theta+\arcsin(\rho/d))$, equivalent to the angle between $\mathbf n$ and $\mathbf v$ exceeding $90^\circ+\theta+\arcsin(\rho/d)$. It is applied only to *closed* parts (solids), and not to glass: on an open or transparent surface the back face is visible.

**Occlusion cull** — Each node has a mask of 16 × 16 directions (octahedral, in asset space) from which something of its subtree is visible. The direction $\mathbf v$ from the node to the eye is computed, its position on the grid $\mathrm{oct}(\mathbf v)\cdot16-0.5$, and the four neighbouring cells a bilinear interpolation would use; if all four are closed, the node is not seen from there. It is valid only with the eye **outside** the asset box: the masks were computed for directions coming from outside (inside the car, for instance, they say nothing).

*References:* on normal cones: Shirman, L. A., Abi-Ezzi, S. S. (1993), "The Cone of Normals Technique for Fast Processing of Curved Patches", Computer Graphics Forum; athenea `shaders/athenea/lod/occlusion_mask.slang`, `shaders/athenea/lod/lod_tree_cut.slang`.
:::

athenea's codebook for the car has 8192 entries (84 % of the ideal, with no false positives), and the v8 codebook has 8182. With quantised centres, occlusion is tested over the box of possible centres (split by orthant, where both octahedral coordinates are monotone). Over the car's 10.7M nodes, at athenea's cameras and 24 random eyes, this hid 0 nodes that the exact centre keeps, at a cost of at most 0.06 % of the hidden nodes being kept. With every cull on, the drawn count is −51 % against every cull off, at 36.9 dB between the two images.

::: {.nota}
**The occlusion codebook** — The occlusion masks are also quantised to a dictionary, but with a different rule from the open directions: each entry is the **OR** of its members. An entry thus never closes a direction that some member has open (no false positives: nothing visible is ever hidden), at the cost of opening extra directions (it hides less than the ideal). "84 % of the ideal" means it hides 84 % of the nodes that the exact per-node masks would hide.

**Quantised centres** — In the compact traversal the node's centre is quantised (§7.5.3), so the exact direction to the eye is uncertain. The box of possible centres is tested: split by octants (in each one, the octahedral coordinates are monotonic in each component, so the 8 corners give their range), the node is hidden only if every cell that any direction from that box could read is closed. The test loses little (it keeps at most 0.06 % of the nodes the exact centre would hide) and hides nothing the exact one keeps (0 out of 10.7M nodes).

**−51 % drawn, 36.9 dB** — With both culls, half the drawn nodes disappear; the image changes by 36.9 dB with respect to drawing everything. It is not 0 because the hidden gaussians contribute something at the edges (semi-transparency, silhouettes), and the masks have a margin of one cell.

*References:* athenea `apps/athenea/src/CmdDecimate.cpp` (`--occlusion-mask`, `--occlusion-group`), athenea `modules/usd/src/Occlusion.cpp`.
:::

### 7.5.2 The compact tier and its versions

The tier-4 section `CMPC` holds the 40 GPU bytes plus the traversal's bytes, so a phone reads none of tiers 1–3. The tables shared by a piece (`CMPT`: dictionary, materials, basis) and by an asset (`.cmpa`: basis, one dictionary, one material table, and the occlusion codebook with the asset's box; identified by FNV-1a 64) are read once.

::: {.nota}
**`CMPC`, `CMPT` and `.cmpa`** — Three levels of compact-path data: `CMPC` is the per-node records (tier 4, in each page); `CMPT` a part's tables (shadow dictionary, materials, CPCA basis) after the page table; `.cmpa` a separate file with the tables shared by all the asset's parts (basis, one dictionary, one material table and the occlusion codebook with the asset box).

**FNV-1a 64** — A non-cryptographic hash function (Fowler–Noll–Vo): $h_0 = \texttt{0xcbf29ce484222325}$ and, for each byte $b$, $h \leftarrow (h \oplus b)\cdot\texttt{0x100000001b3} \bmod 2^{64}$. Fast, table-free, and enough to identify a file's contents: each part stores the id of the asset table it was written with, and the reader checks that the loaded `.cmpa` is that one (otherwise the indices into the dictionary or the basis would point at something else).

*References:* Fowler, G., Noll, L. C., Vo, K.-P., Eastlake, D., "The FNV Non-Cryptographic Hash Algorithm", IETF draft-eastlake-fnv.
:::

| `CMPC` | bytes a node (file) | traversal | change |
|---|---|---|---|
| v1 | 64 = 40 + 24 | LoD tree's 4 words + `LODE`'s 2 | first compact pages; paged = resident bit for bit |
| v2 | 56 = 40 + 16 | 16 B file, 14 B WASM | `CMPT` v2 and the `.cmpa` asset table; same cut node for node; near-view download 23.7 → 16.8 MB, tables 3.62 → 0.97 MB |
| v3 | 64 = 40 + 24 | +cone word, 13-bit occlusion index; 30 B in, 20 B WASM | back-cull and occlusion on by default; only the pages the traversal asks for are planned |
| v4 | container `CMP4` | 12-byte quantised records | payload as **raw-deflate runs of 64 nodes**, plane after plane; node pool; tier 4 +10 % (314 → 344 MB) |

::: {.nota}
**The `CMPC` versions** —

- **v1** (64 B = 40 GPU + 24 traversal): the traversal reuses the 4 words of Spark's LoD node and the 2 of `LODE`. A proof of concept: the paged cut is bit-for-bit that of everything resident.
- **v2** (56 B): the traversal shrinks to 16 B (centre as a correction in f16 order on top of the compact position, children's relative page, children's position and count, $\varepsilon$ and $\rho$), 14 B in the WASM tree. The tables move to the `.cmpa`: the 516 KB basis was repeated in every part (3.1 of the 3.6 MB of tables for six parts). Hence 3.62 → 0.97 MB of tables and 23.7 → 16.8 MB of close-up download.
- **v3** (64 B): adds the cone word (for the back-cull) and the occlusion index (13 bits), which turns both culls on by default. And the streamer requests only the pages the traversal really asks for (pruned ones cost no residency).
- **v4** (`CMP4` container): the traversal records quantised to 12 B and the payload in runs of 64 nodes compressed separately. Tier 4 grows by 10 % (small runs compress worse than a whole page: DEFLATE has less context), but it can be downloaded and evicted with a granularity of 64 nodes.
:::

The **v4 page** is a container: a header, each run's end offset, the gzipped traversal planes, then each run of 64 nodes compressed by raw deflate. Runs can be fetched, kept and evicted one by one, so the CPU keeps a page's payload compressed and the GPU keeps records only for the **drawn** cut. The node pool's census motivated this: with whole pages resident, only about 26 % of each page's nodes were drawn, and smaller pages did not help (residency/cut 4.54× at $P=3000$, 4.46× at $P=500$) (Figure 18).

::: {.nota}
**The v4 page from the inside** — A header (`"CMP4"`, nodes, nodes per run = 64, traversal size, number of runs), the end of each run (so a single one can be requested with Range), the traversal planes in gzip (the CPU always needs them) and each 64-node run with its GPU planes in *raw deflate* (DEFLATE without gzip's header or CRC: 18 bytes less per run, which counts for small runs). The player inflates them with `fflate` in a worker.

**Why smaller pages were not enough** — The census showed that only ~26 % of a resident page's nodes were drawn. Lowering $P$ from 3000 to 500 barely changed the resident/cut ratio (4.54× → 4.46×): the overhead is not one of granularity, it is structural (the cut's ancestors, the nodes pruned by the culls and those that fall below the cut are in the same pages as the drawn ones). The solution was to separate *what decides the cut* (the traversal, small, always on the CPU) from *what is drawn* (the 40 B, only for the cut's nodes, in a GPU node pool).

*References:* RFC 1951 (raw DEFLATE) and RFC 1952 (gzip wrapper).
:::

![Figure 18. The compact format: an asset table, `.athc` v3 files whose tier 4 holds v4 pages of traversal records and 40-byte payload records, and the clustered-PCA transfer basis kept on the GPU.](figures/fig-compact.svg)

::: {.nota}
**Figure 18** — It summarises the compact format's data hierarchy: the asset table (`.cmpa`) downloaded once; the `.athc` v3 files, whose tier 4 holds the v4 pages (header, traversal records, payload runs of 40 B per node); and, resident on the GPU, the transfer's CPCA basis (504 words per cluster) with its projections (144 per cluster).
:::

### 7.5.3 Traversal records

The WASM traversal's per-node cost went down in steps, each producing the same cut:

::: {.nota}
**The traversal in WASM** — The traversal (deciding the cut) runs on the CPU, in Rust compiled to WebAssembly, over a compact copy of the tree: it is not image arithmetic but a tree walk with local decisions, which on the CPU is simple and requires reading nothing back from the GPU. Every byte per node of that copy is phone memory that counts against the budget, hence the race to reduce it.
:::

| stage | bytes a node | how |
|---|---:|---|
| Spark `LodSplat` | 24 | full tree |
| `CMPC` v2 / v3 in WASM | 14 / 20 | f16 centre as an f16-order step from the compact position (exact), $\varepsilon,\rho$ f16, children |
| v4 `QuantNode` | 12 | centre $3\times10$ bits in the page's box of centres; $\varepsilon$ 8 and $\rho$ 7 bits of log code over the page's range; $\theta$ 6; axis 5+5; occlusion 13; children as place + page delta |
| class records | 5, 7, 9 or 11 | 2 bits pick leaf/merge × cone/no cone; 9-bit centres; **9-bit index into the page's occlusion palette** (511: none); a byte offset every 32 nodes |

::: {.nota}
**The records table** —

- **Spark's `LodSplat` (24 B)**: Spark's generic node (f32 centre, size, children...).
- **CMPC v2/v3 (14/20 B)**: the centre is not stored in full: the merge has a compact position (u16 in the page box, already present in the payload) and the f16 centre the traversal needs is encoded as a small number of *steps in f16 order* from that position's f16. "f16 order" is the trick of interpreting a half's bits as a monotonic integer (positives as they are; negatives inverted), so that $\pm k$ steps are the $k$ neighbouring halves: the correction is exact and fits in few bits.
- **`QuantNode` (12 B = 96 bits)**: centre $3\times10$ bits in the page's box of centres; $\varepsilon$ in 8 bits and $\rho$ in 7, as logarithmic codes over the page's range (decoded with a per-page table); $\theta$ in 6 bits; cone axis 5+5 bits; occlusion index 13 bits; children as position + page delta.
- **Records by class (5, 7, 9 or 11 B)**: 2 bits say whether the node is a leaf or a merge and whether it has a cone; each class stores only what it uses. A leaf without a cone needs no $\varepsilon$, $\rho$, children or cone: 5 bytes (27-bit centre + 9 occlusion + 2 class...). To reach node $i$ without walking the previous ones, a byte offset is stored every 32 nodes: random access with a walk of at most 31 records.

**The per-page occlusion palette** — Instead of a 13-bit index into the global codebook, each page stores the list of distinct entries its nodes use (few, because they are neighbours) and each node a 9-bit index into that list (511 = no mask).
:::

Rounding only refines more and culls less: $\varepsilon,\rho,\theta$ are rounded up, a merge's axis error is added to its $\theta$, and the page's largest centre error is added to every sphere. Across the whole car, 1.15M resident nodes take 10.4 MB as class records, against 13.8 MB as 12-byte words. The page budget assumes 8 B a node. A further lossless step pointed each page's merges to their children's page through a **per-page palette** of 9 bits. This kept 436 nodes in full form instead of 181k on the 1280 car, and removed a 2.5 MB WASM map.

::: {.nota}
**Conservative rounding** — Quantising the bounds must not make the traversal refine *less* than it should (that would leave visible errors larger than $\tau$) nor prune anything visible. Hence: $\varepsilon$, $\rho$ and $\theta$ are rounded up (larger bounds → more refinement, less culling); the error of a merge's quantised axis is added to its $\theta$ (the quantised cone still contains the normals); and the page's largest centre error is added to every sphere (the quantised sphere still contains the subtree). The result is the same cut or a slightly finer one, never a worse one.

**The figures** — 1.15M resident nodes: 10.4 MB by class (≈9 B/node) against 13.8 MB in 12 B words. The page budget assumes 8 B per node as a planning estimate.

**The child-page palette** — As with occlusion: a merge points at its children's page with a 9-bit index into a per-page list of child pages. Before, a child in a distant page (large delta) did not fit and the node was stored in full form (20 B); with the palette, only 436 nodes (against 181k) stay that way in the 1280 car, and an auxiliary 2.5 MB map in WASM disappears.
:::

### 7.5.4 Synthetic multi-level roots

A page-0 slot holds 3072 nodes, but athenea's car5120 forests keep thousands of lone roots on purpose (merges never cross parts or athenea's guards): fine5 has 29 213 and fine6 21 065. Past a fan-out (2048), the packer cuts the roots, in Morton order of their centres, into runs, each the children of a synthetic node, and repeats until page 0 can hold the rest. Each synthetic node has infinite error (always refined), a sphere around its children, no cone and no occlusion index, so the cut is the flat forest's (tested at 5 eyes × 5 values of $\tau$). fine5 went from 10 files to 3 and fine6 from 8 to 1.

::: {.nota}
**Synthetic roots** — athenea's forests have thousands of roots because merges never cross the boundary between parts (a merge of two different parts would mix incompatible materials and transfers) nor athenea's guards (criteria that prevent merging, for example, gaussians with opposite normals). But page 0 must contain all the roots and has a slot of 3072 nodes (and a child's position, 12 bits, allows up to 4096). The solution: build an artificial hierarchy on top. Beyond a fan-out of 2048, the roots are sorted by Morton order (so that the groups are spatially compact and their spheres small), cut into runs, and each run hangs from a synthetic node; this repeats until what remains fits in page 0.

**Why the cut does not change** — A synthetic node has infinite error: $P_n = \infty > \tau$ always, so it is always refined and never drawn; with no cone or occlusion index, it is never pruned. The resulting cut is exactly that of the flat forest (checked with 5 eyes × 5 values of $\tau$). The practical effect: fewer files per part (fine5 from 10 to 3, fine6 from 8 to 1), because before the parts had to be split into several files (`--split-roots`) for their roots to fit.
:::

## 7.6 Streaming and the per-frame cut

The reader (`CompactBandStreamer`) reads only the page tables at open. After each traversal:

1. The **needed set** comes from each tree's page table alone, with the eye in the tree's object space (the car's turn from athenea's Z-up stage) and foveated.
2. The **missing pages** are requested in order of the error a page removes per byte it costs; ties go to the nearer page. Consecutive pages are merged into one Range request (up to 8 MiB), within the pool's free room.
3. The pool **evicts** pages that are not needed from the finest band down. Page 0 and any page a resident page depends on are never evicted.
4. **$\tau$** rises until the needed set fits (`tauToFit`). The current version finds, by bisection, the lowest $\tau$ that fits, which stopped the near-view oscillation (4.61–5.76 → 5.16).

::: {.nota}
**The streamer's cycle** —

1. **Needed set**: the page test (§7.5.1) over each tree's page table, with the eye transformed into the tree's object space (the trees are in the file's frame; athenea's stage is Z-up and three.js is Y-up, hence "the car's rotation"). **Foveation** (from Spark) lowers the detail demanded away from the centre of the view: the effective $\tau$ grows with the angle to the camera axis.
2. **Priority**: each missing page is scored by the error it removes per byte it costs (the greedy criterion of a fractional knapsack: maximum benefit per unit of capacity); ties, the nearest one. Consecutive pages are grouped into one Range request of up to 8 MiB (fewer requests and less latency), within the pool's free space.
3. **Eviction**: first the unneeded pages of the finest bands (the coarse ones are those needed again soonest). Never page 0 nor a page a resident page depends on (a page whose nodes are parents of resident nodes: without it, the children would be orphaned).
4. **`tauToFit`**: if the needed set does not fit in the pool, $\tau$ rises. The bisection search for the lowest $\tau$ that fits replaced an incremental adjustment that oscillated (close up, $\tau$ jumped between 4.61 and 5.76; now it stays at 5.16).
:::

A node whose children are absent is drawn as a leaf, or as its parent or children while a run is being decoded. With v4 runs, the cut's own runs and those of its parents and children are always kept, and the rest are dropped first under a run budget. An extrapolated camera (2.5 updates ahead) is traversed too, so its runs are prefetched. Up to 12 Range requests are in flight.

::: {.nota}
**Stand-ins** — If a node of the cut does not yet have its data on the GPU, the closest available thing is drawn: its parent (coarser) or its children (finer) if present; a node whose children have not arrived is drawn as a leaf. The image never has holes, only temporarily lower detail.

**Run budget by class** — The CPU keeps, compressed, the runs of the cut's nodes, of their parents and of their children (those that will be needed if the camera moves a little closer or further away); the rest are evicted first.

**Extrapolated camera** — The camera predicted 2.5 updates ahead (linear extrapolation of the motion) is also walked and its runs requested: speculative prefetching that hides network latency during orbits.

**12 requests in flight** — Several simultaneous requests hide each one's latency. With HTTP/2 or HTTP/3 they are multiplexed over one connection; with HTTP/1.1 browsers limit themselves to about 6 connections per host.

*References:* RFC 9113 (HTTP/2).
:::

The work is split across the main thread, the WASM traversal and two workers (Figure 19):

| where | what | measured |
|---|---|---|
| loader worker (WASM) | traversal over all trees, streaming the cut in chunks; stack instead of heap when the budget does not bind | Camera, $\tau=3$: 128 → 40 ms (same cut) |
| node worker (`NodeCutter`) | cut to slots, run inflation (~0.3 µs a node), record packing, draw list with substitutes, prefetch | JS packing 32.8 → ~9 ms; orbit max / p99 from 31–35 / 26–31 to 17.5–19.0 / 17.4–18.7 ms |
| main thread | Range fetches, writes of records and draw lists (adopted, no GC) | — |
| GPU compute | record moves when the pool shrinks (one dispatch, a thread a word) | first frame after a still frame 88–100 → 34 ms |

::: {.nota}
**The threads** — JavaScript in the browser has a main thread (the page's: events, DOM, submitting work to the GPU) and *Web Workers* (separate threads that communicate by messages). A long pause on the main thread shows as a dropped frame, so heavy work goes to workers:

- **Loader worker (WASM)**: the traversal of all the trees. It uses a stack (depth-first walk, no priority order) when the budget does not bind, and a heap (priority queue by error) only when it has to cut by budget; both give the same cut if the budget does not act. The cut is sent in chunks so that processing can start before it finishes. 128 → 40 ms in the Camera view with $\tau=3$.
- **Node worker (`NodeCutter`)**: assigns the cut's nodes to pool slots, inflates runs (~0.3 µs per node), packs records, builds the draw list with stand-ins and manages prefetching. Packing in JS went from 32.8 to ~9 ms; the frame peaks on the orbit, from 31–35 ms (max) to 17.5–19.0 ms.
- **Main thread**: the Range requests (the worker says which) and writing records and lists into GPU buffers. "Adopted, no GC": the arrays arrive as *transferables* (ownership of the `ArrayBuffer` is transferred without copying it) and the previous list is returned to the worker for reuse, so no garbage is created for the collector to clean up in the middle of an orbit.
- **GPU compute**: when the record pool shrinks, a dispatch with one thread per word moves the records to their new positions; before, this was done on the CPU and the first frame after a still frame took 88–100 ms, now 34.

*References:* HTML Living Standard, "Web workers" and "Transferable objects".
:::

![Figure 19. The player’s threads: Range fetches and the page pool on the main thread, the traversal in a WASM worker, the cut packed into slots in a second worker, and five GPU passes a frame; the GPU time drives the τ floor.](figures/fig-player.svg)

::: {.nota}
**Figure 19** — The division of work over time: the main thread handles network and pool; the traversal and the packing run in parallel in two workers; the GPU runs five passes per frame (§7.7). GPU time is what bounds $\tau$ from below: one cannot ask for more detail than the GPU draws at the target rate (§7.10).
:::

On a fast orbit of 120° at 1° a frame on the emulated phone, the share of the cut missing from a frame fell from 26 % with the first v4 streamer to 4.3–6.4 %, which is 0.24–0.30 % of the screen area. Frames stayed at 60 fps with a maximum of 17.3–19.6 ms. The cut settled 2.6–3.1 s after load.

::: {.nota}
**The test orbit** — 120° at 1° a frame: at 60 fps, two seconds of continuous motion, a hard case for streaming because every frame uncovers new areas. "Share of the cut missing" is the cut's nodes whose record is not yet on the GPU (they are drawn with stand-ins); 4.3–6.4 % of nodes, but only 0.24–0.30 % of the screen area, because what is missing are fine, small nodes. 60 fps with a maximum of 17.3–19.6 ms: the odd frame goes slightly over the 16.7 ms period. Settling 2.6–3.1 s after the load: the time until the cut stops changing.
:::

## 7.7 The draw

The compact path (`CompactRelitPath`, `compact.slang`) runs five passes a frame:

1. **Project** (one thread a cluster, only when the light changes): the per-cluster projections of §7.2.2.
2. **Cull** (one thread a listed node): frustum test; a 24-bit radial-distance key and the index appended with an atomic.
3. **Sort**: GPU radix sort of the drawn keys, back to front (6 passes over 24 bits).
4. **Shade** (one thread a drawn slot, binned by cluster): decode, relight (§7.8), project. A 16-byte record is written into the sort's own buffers: centre as u16 at 1/16 px, $\sigma_1$ (half) with the major-axis angle, $\sigma_2$ (half) with $\alpha$ (half; a negative sign marks LoD opacity), and colour in RGB9E5.
5. **Draw**: instanced quads, premultiplied over an rgba16float target.

::: {.nota}
**Compute versus render pipelines** — In WebGPU a *compute pipeline* runs a kernel over a grid of threads (`dispatchWorkgroups`), without rasterising: it reads and writes buffers. A *render pipeline* is the classic graphics pipeline (vertex shader → rasteriser → fragment shader → blending onto a target). The first four passes are compute; the fifth is render.

**The five passes** —

1. **Project**: the per-cluster projections of §7.2.2, only when the light changes.
2. **Cull**: one thread per node of the draw list; a frustum test (is the sphere inside the view pyramid?). Survivors reserve a position with an `atomicAdd` on a global counter and write their 24-bit key (radial distance to the eye, not z depth: radial order does not change when the camera rotates, which avoids order *popping*) and their index.
3. **Sort**: an LSD (least significant digit first) *radix sort* on the GPU: 6 passes of 4 bits over the 24-bit key. Each pass is a stable *counting sort* by one digit (histogram, prefix, scatter). Back to front, because "over" *blending* of semi-transparent splats requires that order.
4. **Shade**: one thread per drawn slot; it decodes the compact record, computes the relight (§7.8) and projects the gaussian. It writes a 16 B record into the sort buffers: centre in u16 at 1/16 pixel (with a 1024 px offset: it covers −1024 to 3072 px), $\sigma_1$ (half) with the angle of the major axis, $\sigma_2$ (half) with $\alpha$ (half; its negative sign marks LoD opacity), and the colour in **RGB9E5** (three 9-bit mantissas with a shared 5-bit exponent: HDR colour in 32 bits, WebGPU's `rgb9e5ufloat` format).
5. **Draw**: one quad per splat (instanced drawing: 4 vertices, one instance per splat), colours *premultiplied* by alpha and blended onto an `rgba16float` target (half per channel, HDR).

**Grouped by cluster** — See §7.8 ("The reflected field").

*References:* [46] (WebGPU, §"Compute Pipelines" and §"Render Pipelines"); Merrill, D., Grimshaw, A. (2011), "High Performance and Scalable Radix Sorting", Parallel Processing Letters.
:::

**Projection (EWA).** With $W$ the view rotation, $R,S$ the splat's rotation and scale, and $J$ the perspective Jacobian at the view-space centre $(x,y,z)$:

$$
J = \begin{pmatrix} f_x/z & 0 & -f_x x/z^2\\ 0 & f_y/z & -f_y y/z^2\end{pmatrix},\qquad
\Sigma_{2D} = J\,W R S\,(J W R S)^{\!\top} = \begin{pmatrix} a & b\\ b & d\end{pmatrix}.
$$

::: {.nota}
**EWA splatting** — Zwicker et al. [3]: a 3D gaussian with covariance $\Sigma = RSS^\top R^\top$ ($R$ rotation, $S$ diagonal scale) is projected to a 2D gaussian by approximating the perspective projection with its linearisation (the *Jacobian* $J$) at the splat's centre. $W$ takes world to view. The 2D covariance is $\Sigma_{2D} = JW\Sigma W^\top J^\top$, which is what the formula writes as $JWRS(JWRS)^\top$. The matrix $J$ is the derivative of $(x,y,z)\mapsto(f_x x/z, f_y y/z)$, with $f_x,f_y$ the focal lengths in pixels. It is the projection of 3D Gaussian Splatting [1]. The affine approximation fails for large gaussians very close to the camera or at the edges of a wide field, where perspective is not locally linear.

*References:* [3], [1].
:::

**Opacity compensation.** The anti-aliasing blur adds $\beta = 0.3\ \text{px}^2$ to the diagonal, and the opacity is scaled so the splat's mass is kept:

$$
\Sigma' = \Sigma_{2D} + \beta I,\qquad \alpha' = \alpha_0 \sqrt{\det\Sigma_{2D}/\det\Sigma'} .
$$

::: {.nota}
**The anti-aliasing blur and the compensation** — A gaussian smaller than a pixel produces *aliasing* (it flickers as it moves). Adding $\beta I$ to the covariance is equivalent to convolving it with a gaussian filter of variance $\beta=0.3$ px² (the value used by Spark and by the original 3DGS). But widening it as is increases the splat's mass (its integral) and makes it more visible. The integral of $\alpha\,e^{-\frac12\mathbf x^\top\Sigma^{-1}\mathbf x}$ is $2\pi\alpha\sqrt{\det\Sigma}$; to keep it when going from $\Sigma_{2D}$ to $\Sigma'$, $\alpha$ must be scaled by $\sqrt{\det\Sigma_{2D}/\det\Sigma'}$. It is Mip-Splatting's compensation [4].

*References:* [4], [1], [3].
:::

A merged node with LoD opacity $\alpha_0 > 1$ is drawn with $\alpha_0 \leftarrow \min(4\alpha_0 - 3, 5)$, and its quad is widened by $0.7(\alpha_0-1)$ deviations. A faint splat ($\alpha' < 4\,\alpha_{\min}$) keeps its mass $\alpha_0\sqrt{\det\Sigma_{2D}}$: its blur is chosen so its opacity reaches the faint target, or it is drawn at that target with probability $\alpha'/(4\alpha_{\min})$ by a stable hash dither. A thin glass sheet's floor of 0.1 is applied **after** the blur and faded by its area on screen. Applied before the blur, it had made the headlight cover 1.63× too bright. The fragment evaluates $\alpha = \alpha'\,e^{-z^2/2}$ inside the cut radius, discards under $\alpha_{\min}$, and adds the per-pixel sharp lobes (§7.8).

::: {.nota}
**LoD opacity** — Spark marks a merge with $\alpha_0 > 1$: its "excess" opacity indicates that it must be drawn wider and more opaque, to cover its subtree's mass. The player converts it with $\alpha_0\leftarrow\min(4\alpha_0-3,5)$ and widens the quad by $0.7(\alpha_0-1)$ standard deviations (Spark's constant).

**Faint splats** — With $\alpha_{\min}=0.5/255$ (below it, a fragment does not change an 8-bit target), a splat whose $\alpha'$ falls below $4\alpha_{\min}$ would be drawn with an alpha that the blending's quantisation mistreats. Two remedies that keep the mass $\alpha_0\sqrt{\det\Sigma_{2D}}$: (a) choose the blur $b$ so that the compensated opacity reaches exactly the target $4\alpha_{\min}$ (solving a quadratic equation in $b$); or (b) if even then it does not reach it, draw it at opacity $4\alpha_{\min}$ with probability $\alpha'/(4\alpha_{\min})$. In expectation, $p\cdot4\alpha_{\min} = \alpha'$: it is *stochastic transparency*, unbiased. The "stable hash" (a bit-mixing function on the gaussian's index) makes the decision the same in every frame: no flicker.

**Glass sheet floor** — Thin glass sheets have a minimum alpha of 0.1 so that they are visible; applying it *before* the blur made the sheet, after compensation, worth more than intended (the headlamp cover 1.63× too bright). Applied afterwards, and faded by its screen area (a one-pixel sheet should not weigh like a large one), it comes out right.

**The fragment** — For each pixel of the quad, $z^2 = \mathbf d^\top\Sigma'^{-1}\mathbf d$ (Mahalanobis distance to the centre) and $\alpha = \alpha' e^{-z^2/2}$; outside the cut-off radius ($z^2>8$, i.e. $\sqrt8\approx2.83$ deviations, Spark's `maxStdDev`) or below $\alpha_{\min}$ it is discarded. Then it adds the per-pixel sharp lobes (§7.8).

*References:* Enderton, E., Sintorn, E., Shirley, P., Luebke, D. (2010), "Stochastic Transparency", I3D; [4], [11].
:::

A thread per drawn slot reuses the sort buffers as records, so nothing else is per-splat. The compact path has no relit records (84 B), no pixel records (48 B), no accumulator (32 B) and no ext splats (32 B). On the phone it also has no depth texture: nothing reads it. A still view re-runs only the draw. Writing directly to 8 bits saved no GPU time (Apple blends in tile memory) and changed the image by up to about 100 codes in 1–3 % of channels, so the target stays rgba16float.

::: {.nota}
**What does not exist on the compact path** — The desktop relight stored per gaussian a relit record (84 B), per pixel records (48 B) and an accumulator (32 B), and Spark's *ext splats* (32 B): all of that goes away because the shade writes the draw record directly and the fragment composites with no intermediate state. No depth texture (nobody reads it: splats are sorted, not z-tested). With the view still, only the draw pass is repeated.

**Why an 8-bit target did not save time** — Apple GPUs are *tile-based deferred renderers* (TBDR): blending happens in the tile's on-chip memory and the target is only written to memory at the end. Blending bandwidth does not depend on the format, so moving to 8 bits gained no time; and since 8-bit blending rounds at every splat, it changed the image (up to 100 codes out of 255 in 1–3 % of the channels, through accumulated rounding over hundreds of semi-transparent layers). `rgba16float` accumulates with 11 mantissa bits without those errors.

*References:* Apple, "Tailor your apps for Apple GPUs and tile-based deferred rendering" (Metal documentation).
:::

## 7.8 Relighting on the phone

The shade evaluates athenea's `relitByDome` for one dome. The code is a verbatim copy with the field reads taken from registers, because WebGPU cannot bind one buffer as both read and write in a dispatch. The kept terms come from the projections; the reflected field (16 coefficients × RGB) is read along the mirror. The following refinements were ported or added, and each was measured.

::: {.nota}
**`relitByDome`** — The engine function that computes the radiance of a gaussian with transfer under a dome (a *dome light*: an HDR environment map) plus the sun: diffuse terms by SH dot product, specular through the reflected field, the sun through the open directions, glass transmission. The player does not reimplement it: it copies it literally (Slang compiles to WGSL just as it does to Metal), and only changes where it reads the data from.

**Why the reads come from registers: binding aliasing in WebGPU** — WebGPU validates, for each dispatch (its *usage scope*), that the same buffer is not used at once as writable (`storage` with `read_write`) and in any other binding (read-only, uniform or another storage). If a subresource has a writable usage in the scope, that usage must be exclusive; otherwise the dispatch is a validation error. In Metal or Vulkan this is allowed (at the programmer's risk); in WebGPU the rule exists to guarantee portability and the absence of undefined data races. Since the shade writes into the sort buffers and reads record fields that share a buffer, the copied code is adapted: the reconstructed transfer field is passed in local variables (GPU registers), not as buffer reads.

**Conserved terms and reflected field** — The conserved ones (direct and indirect diffuse) come from the per-cluster projections (§7.2.2); the reflected field is 16 SH coefficients × RGB describing what is seen reflected in each direction, evaluated along the mirror direction.

*References:* [46] (WebGPU, §"Resource Usages" and "Synchronization": *usage scope* and the exclusive writable usage rule); athenea `shaders/athenea/splat/splat_relight.slang` (`relitByDome`).
:::

**The reflected field.** It is 42 % of the shade, and the cost is arithmetic, not memory. Two changes cut it: shading binned by cluster (a counting sort of the slots per cluster, with per-workgroup histograms), −6 to −9 %; and the field limited to degree 1 (4 coefficients) when a footprint is under 3 px, −23 % of the shade at 215k drawn. Against the path trace, the change in the door, sill, bumper and hood is under 0.01 dB.

::: {.nota}
**42 % of the shade, arithmetic** — Evaluating 16 SH × RGB in one direction means computing the 16 basis polynomials and 48 multiply-adds, plus rebuilding the 48 coefficients from the basis (8 axes × 48 + mean). It is computation (ALU), not memory reads: reducing bytes does not speed it up; one has to reduce operations or improve their coherence.

**Shade grouped by cluster: a counting sort with per-workgroup histograms** — On a GPU, threads execute in SIMD groups (32 on Apple, a *simdgroup*); if the 32 threads read the same cluster's basis, the reads are served from cache (or broadcast), and if each reads a different cluster, they are 32 scattered reads of hundreds of words. That is why the drawn slots are reordered by cluster with a *counting sort*, in three passes: (1) `BinCount`: each workgroup (256 threads) counts its slots per cluster in shared memory (`groupshared`, with local atomics) and adds its histogram to the global one with **one atomic per cluster and workgroup**, not one per slot (much less contention); (2) `BinScan`: a prefix sum of the histogram (where each cluster starts); (3) `BinScatter`: each slot is written to its cluster's position. Order within a cluster does not matter because each slot is shaded in its own place. Gain: −6 to −9 % of the shade.

**Field limited to degree 1** — SH of degree $\le l$ have $(l+1)^2$ coefficients: degree 1 is 4 (constant + 3 linear), degree 3 is 16. A footprint smaller than 3 px cannot show the fine angular variation of the reflection within itself (a sampling argument: there are not enough pixels to represent that frequency), so only 4 of the 16 coefficients are evaluated: −23 % of the shade with 215k drawn and less than 0.01 dB of change against the path trace in the measured regions.

*References:* Cormen et al., *Introduction to Algorithms* (counting sort); [16] (SH, degrees and bands).
:::

**Sharp per-pixel lobes.** A coat or polish below the sharp roughness thresholds is not answered once at the centre. The shade subtracts the centre's sky reading and hands the draw a 5-word record: the mirror direction at the centre (unorm16 × 2), its steps one pixel right and one pixel down (f16 pairs), the coat weight and both roughnesses as square roots, and the polish weight in RGB9E5. The fragment then reads the sky along $\mathrm{oct}^{-1}(\mathbf a_0 + \mathbf s_x\,\delta x + \mathbf s_y\,\delta y)$. The roughness read is built in three steps:

::: {.nota}
**Per-pixel sharp lobes** — The shade computes once per gaussian, at its centre. For the diffuse that is enough (it varies slowly), but the reflection of a clear coat or a low-roughness polish changes within the splat's footprint (the curved surface turns the mirror direction from one pixel to the next). That is why, below a roughness threshold, the shade *subtracts* its sky reading at the centre and leaves the sharp part to the fragment.

**The 5-word (20 B) record** — The mirror direction at the centre, in octahedral coordinates $\mathbf a_0$ (unorm16 × 2); its screen derivatives, how much the octahedral coordinate changes one pixel to the right $\mathbf s_x$ and one down $\mathbf s_y$ (f16 pairs); the clear coat weight and its two roughnesses (stored as square roots, which give more precision at low roughness); and the polish weight in RGB9E5. The fragment rebuilds the direction with a first-order Taylor expansion, $\mathrm{oct}^{-1}(\mathbf a_0+\mathbf s_x\delta x+\mathbf s_y\delta y)$, with $(\delta x,\delta y)$ its offset from the centre in pixels, and reads the prefiltered sky in that direction. Linearising in octahedral space is cheap and continuous within one face of the octahedron.
:::

- *Unwidening.* The bake widened the stored roughness by Toksvig, $r'^4 = r^4 + 2s^2$ with $s$ the normal spread. Since the per-pixel turn already accounts for that spread, the lobe reads $r = \max(r'^4 - 2s^2, 0)^{1/4}$.
- *Fold rule.* With $t = 2s$ the turn over $\pm2\sigma$ and $w = \mathrm{smoothstep}(0.7, 1.5, t)$, the pixel's turn is scaled by $1-w$ and the lobe widens, $\alpha' = \sqrt{\alpha^2 + w\,(t/2)^2}$, held under 0.5.
- *Pixel spread.* The normal turns by $\mathbf d_x,\mathbf d_y$ across a pixel. That box of slopes, doubled for the mirror, widens the GGX $\alpha = r^2$:
$$
\alpha^2 \leftarrow \alpha^2 + \frac{2\left(\lVert\mathbf d_x\rVert^2+\lVert\mathbf d_y\rVert^2\right)}{12} = \alpha^2 + \frac{\lVert\mathbf d_x\rVert^2+\lVert\mathbf d_y\rVert^2}{6}.
$$

::: {.nota}
**Roughness and GGX** — In the GGX distribution (Walter et al. [24]), the parameter $\alpha$ is the lobe's width in microfacet slopes, with $\alpha = r^2$ ($r$ being the materials' "perceptual" roughness). Widening a lobe by a spread of normals is done by adding variances: $\alpha^2 \leftarrow \alpha^2 + \sigma^2_{\text{slope}}$ (a gaussian approximation of the slopes, the basis of the geometric *specular anti-aliasing* of Kaplanyan et al. 2016 and Tokuyoshi–Kaplanyan 2019).

**Unwidening** — The bake applied Toksvig [20]: so that a merge of spread normals would not shine like a mirror, it stored a widened roughness $r'^4 = r^4 + 2s^2$ (i.e. $\alpha'^2 = \alpha^2+2s^2$, with $s$ the spread of normals). But the per-pixel path already represents that turning of the normal explicitly (the reflection moves within the splat); counting it twice would blur. It is inverted: $r = \max(r'^4-2s^2,0)^{1/4}$.

**Fold rule** — A *fold* is a sharp crease in the surface (a crisp edge of the bodywork). There the linearisation of the mirror direction stops holding (it turns too much across the splat). With $t=2s$ (the turn over the gaussian's $\pm2\sigma$) and a smooth weight $w=\mathrm{smoothstep}(0.7,1.5,t)$ (0 for small turns, 1 for large ones, with a $C^1$ transition), the per-pixel turn is reduced to $(1-w)$ and that spread is handed back to the lobe as roughness: $\alpha'=\sqrt{\alpha^2+w(t/2)^2}$, capped at 0.5. This switches from a "moving reflection" to a "blurred reflection" where the former is not reliable.

**Pixel spread** — Within a pixel the normal also turns, by $\mathbf d_x$ and $\mathbf d_y$ across its width and height; a pixel averages those normals. A uniform distribution of width $d$ has variance $d^2/12$; the mirror direction turns twice as much as the normal. Adding both screen directions, the lobe widens by $\alpha^2 \mathrel{+}= 2(\lVert\mathbf d_x\rVert^2+\lVert\mathbf d_y\rVert^2)/12$. It is per-pixel *specular anti-aliasing*: without it, high-frequency reflections in strongly curved areas flicker as bright specks.

*References:* [24] (GGX), [20] (Toksvig), [25]; Kaplanyan, A. S., Hill, S., Patney, A., Lefohn, A. (2016), "Filtering Distributions of Normals for Shading Antialiasing", HPG; Tokuyoshi, Y., Kaplanyan, A. S. (2019), "Improved Geometric Specular Antialiasing", I3D.
:::

Without the pixel spread, v7 showed bright specks on folds. At $\tau$ 1 / 3, the specks went from 24 / 27 to 6 / 2 on the sill and from 19 / 25 to 2 / 1 on the door crease; the door went from 34.21 to 35.53 dB. A lobe whose mirror stays within 2 texels of its sky level across $3\sigma$ is read once at the centre (the draw −10 to −12 % close up, with no change against the path trace within 0.001 dB).

::: {.nota}
**The figures** — "Specks" are isolated bright points counted in a region (sill = the low area under the doors; door fold = the door's edge), with $\tau$ 1 and 3. The pixel spread brings them down from dozens to a few, and the door improves by 1.3 dB against the path trace (34.21 → 35.53).

**A single read when no more is needed** — If the mirror direction, along the splat's $\pm3\sigma$, does not move more than 2 texels from the sky level its roughness calls for, the lobe is effectively constant over the splat: it is read once at the centre. The draw gets 10–12 % cheaper close up and the image does not change against the path trace (0.001 dB).
:::

**The environment as a filterable atlas.** At first the fragment decoded two levels of four texels each from a storage buffer. The dome's eight prefiltered levels are now copied into one rgba16float atlas: level 0 at the origin, the others down a column to its right (0.9 MB for a base side of 256, 3.1 MB at 512). The atlas is sampled with a hardware bilinear sampler, two samples a lobe. The image is 77 dB of the storage reads. The compact pass at 393 × 852, $\tau=3$ (M5 Pro, wall time) went from 11.8 to 4.0 ms with sharp lobes from 1 px; at 1920, $\tau=1$, from 75 to 17.2 ms. The shade reads its own env texels from the atlas as well, which freed the env buffer (−2.67 MB).

::: {.nota}
**Prefiltered environment** — To read a lobe of roughness $r$ in a direction, a *prefiltered* environment map is used: successive levels convolved with ever wider lobes, so that reading one level is equivalent to integrating the sky under the lobe (Křivánek and Colbert [18]; the *split-sum* of Karis and Frostbite [19]). Eight levels for the dome.

**Filterable atlas versus storage buffer** — Before, the levels lived in a storage buffer and the fragment did bilinear interpolation by hand: 4 texels per level, 2 levels, decoding each one. Now they are copied into an `rgba16float` texture laid out as an atlas (level 0 at the origin, the others stacked in a column to its right) and the hardware bilinear *sampler* is used: one sampling instruction does the 4 reads and the interpolation. Two samples per lobe (two adjacent levels, blended by roughness: a hand-made trilinear, because the atlas is not a hardware mip chain). 77 dB against the manual reads (rounding differences in the hardware filtering).

**The figures** — The compact pass at 393 × 852 with $\tau=3$ goes from 11.8 to 4.0 ms, and at 1920 with $\tau=1$ from 75 to 17.2 ms (4.4×): texture units are dedicated hardware the previous path did not use. The atlas takes 0.9 MB (base side 256) or 3.1 MB (512); the environment buffer (2.67 MB) goes away because the shade also reads from the atlas.

*References:* [18], [19]; [46] (WebGPU `GPUSampler`, `"linear"` filtering).
:::

**polishField and coatFold.** Each material carries its own roughness in pbr bits 26–31, at no cost per node. Where that roughness is $\ge 0.3$, the polish asks its own lobe's cells, and the reflected field answers what they close (the door's ratio to the path trace went from 0.786 to 0.857). Where the surface turns fast, the coat is held: $r_{\text{coat}} = \max(\mathrm{unwidened}(c,s),\ \min(c, 0.3))$. This applies only where the material's roughness is $\ge 0.3$. Applied everywhere, it lit chrome, glass and rim spokes (wheels +63 % relMSE). On the 1280 bake the coat comes out at 0, so the term no longer changes the car.

::: {.nota}
**Per-material roughness in the pbr word** — The material's pbr word has free bits; athenea marks with bit 26 that a roughness is present and stores $\mathrm{round}(r\cdot31)$ in bits 27–31 (5 bits, steps of 1/31). Since the material table is per asset, it costs nothing per node.

**polishField** — Where the material's roughness is high ($\ge0.3$), the *polish* (the base specular layer) is wide and sees a large region of the sky; part of that region may be blocked by the car itself. With `polishField`, the polish consults the open-direction cells of its own lobe (not just the mirror direction) and the reflected field answers the radiance of the closed cells (what the car reflects of itself). The door's brightness ratio against the path trace improves from 0.786 to 0.857 (1.0 would be exact).

**coatFold** — Where the surface turns quickly, the clear coat is clamped to $r_{\text{coat}} = \max(\mathrm{unwidened}(c,s),\ \min(c,0.3))$: unwidening cannot bring the coat's roughness below 0.3 (or below its own value if it was already lower). Restricted to materials with roughness $\ge0.3$: applied everywhere, it brightened chrome, glass and spokes (wheels +63 % relMSE). In the 1280 bake the clear coat comes out at 0, so this term no longer affects the final car.
:::

## 7.9 The contact shadow

A compact set draws without Spark's renderer, so the contact shadow is a pass of its own (`ground-shadow.js`), and it uses gaussians only (no mesh, no AO). athenea's shadow catcher is black gaussians on the ground plane, each with its open-direction cells. They are splatted over the dome before the car: EWA with the same 0.3 px² blur and opacity compensation, black at opacity $\alpha_i = o_i\,\kappa_i$. Black over black needs no sort. The darkening is

::: {.nota}
**Contact shadow** — The contact shadow that anchors an object to the ground; without it the object seems to float. Keeping to the rule that the render uses only gaussians (no ground mesh, no screen-space ambient occlusion), athenea's *shadow catcher* is a layer of black gaussians on the ground plane, each with its open-direction cells (baked with the car on top). They are drawn over the dome before the car, with the same EWA projection, blur and compensation as the car's splats.

**Black over black needs no sort** — The *over* operator with black colour and opacity $\alpha_i$ leaves $C\leftarrow(1-\alpha_i)C$. Applying several layers gives $C\prod_i(1-\alpha_i)$, a product, which is commutative: order does not matter.

*References:* athenea `shaders/athenea/splat/splat_project.slang` (`catcherOpacity`, `kProjectCatcher`); athenea `docs/decisions.md` (section "The ground's shadow as gaussians: a shadow catcher (task PLAY-G)").
:::

$$
\kappa_i = \frac{-\ln r_i}{2\pi},\qquad
r_i = \frac{\displaystyle\sum_{\text{open cells } j} L_{\text{dome}}(\boldsymbol\omega_j)\cos\theta_j\,\Delta\Omega_j \;+\; \text{sun through the cells}}{\text{irradiance of the whole sky above the floor}},
$$

recomputed on the CPU from the dome's 16 harmonics whenever the dome changes. athenea's day catcher (309k gaussians at 1.26 cm) costs 9.9 MB of GPU drawn whole, and thinning it 1 in 4 shows a wavy grain. Instead, `floor_shadow --merge 2` moment-matches 2 × 2 neighbours (centre and covariance weighted by opacity × area, mass kept, cells by majority). The result is 78k gaussians, 0.65 MB gzipped, 2.5 MB of GPU, and the closest to the path trace in every region:

::: {.nota}
**The darkening** — $r_i$ is the ratio between the irradiance reaching the ground point with the car present (summing, over the open cells $j$, the dome radiance $L_{\text{dome}}$ times the cosine and the cell's solid angle, plus the sun if its direction falls in an open cell) and the irradiance that would reach it without the car (the whole sky over the ground). $r_i=1$ with no shadow, $r_i\to0$ in full shadow.

**Why $-\ln r/2\pi$** — The catcher's gaussians form a grid of spacing $d$ with $\sigma = d$; the sum of their falloffs at any point is $\sum_i e^{-\lVert\mathbf x-\mathbf x_i\rVert^2/2\sigma^2}\approx 2\pi\sigma^2/d^2 = 2\pi$. With small opacities, the transmittance is $\prod_i(1-\alpha G_i)\approx\exp(-\alpha\sum_i G_i) = e^{-2\pi\alpha}$. To let exactly the fraction $r$ through: $e^{-2\pi\kappa}=r \Rightarrow \kappa=-\ln r/2\pi$. The constant $2\pi$ (`kCatcherDepth`) replaced a first value of 2.4 that gave shadows 2.2–3 times too deep.

**On the CPU** — When the dome changes, the player recomputes the opacities from the dome's 16 harmonics: a small, infrequent computation (78k gaussians × 256 cells, once per sky change), which the player does in JavaScript. The "no CPU arithmetic" rule belongs to the engine; the player follows it where it matters (per frame).

*References:* athenea `shaders/athenea/splat/splat_project.slang` (`kCatcherDepth = 6.2831853`, comments "HOW MANY OF A CATCHER'S GAUSSIANS A POINT OF IT LIES UNDER" and "The layer leaves exp(-kCatcherDepth a) of what is behind it").

**The catcher reduced by moments** — The day catcher has 309k gaussians at 1.26 cm (9.9 MB of GPU, about 32 B per gaussian). Thinning it to 1 in 4 breaks the grid condition of the previous note (the sum of falloffs is no longer uniform) and a wavy grain appears. Instead, `floor_shadow --merge 2` merges each group of 2 × 2 neighbours into one gaussian that preserves their moments: weights $w_i$ = opacity × area, centre $\mu=\sum w_i\mathbf x_i/\sum w_i$, covariance $\Sigma = \sum w_i\big(\Sigma_i + (\mathbf x_i-\mu)(\mathbf x_i-\mu)^\top\big)/\sum w_i$ (the *moment matching* of Runnalls's gaussian mixture reduction [34]), total mass preserved, and cells by majority. The result, 78k gaussians (0.65 MB gzipped, 2.5 MB of GPU), is a continuous surface with correct coverage.

*References:* [34].
:::

| region (shadow factor) | PT | thinned 1 in 4, 77k | full 309k | merge 2 × 2, 78k |
|---|---:|---:|---:|---:|
| front wheel | 0.607 | 0.565 | 0.568 | **0.594** |
| rear wheel | 0.713 | 0.617 | 0.618 | **0.638** |
| front floor | 0.211 | 0.168 | 0.164 | **0.224** |
| under sill | 0.437 | 0.390 | 0.387 | **0.430** |

::: {.nota}
**The shadow factor table** — Each number is the ratio (ground with car) / (ground without car) measured in a region: 1 with no shadow, 0 black. PT is the path trace. The thinned and the full catcher give almost the same (shadows too dark at the wheels, too light on the front ground); the 2 × 2 merge is the closest in every region (front wheel 0.594 against 0.607; under the sill 0.430 against 0.437). That the merge *improves* on the full catcher, rather than just matching it, suggests that the moment-fitted covariance approximates the continuous coverage better than the original grid of small gaussians.

*References:* athenea `docs/decisions.md` (section "The ground's shadow as gaussians: a shadow catcher (task PLAY-G)", subsection "Measured (M5 Pro, 1920 x 1080, the TX car and its catcher)").
:::

Without the shadow the car floated, and the sill read 1.66× the path trace: it was the hall's floor at infinity seen past the car's edge. With the shadow the sill reads 1.08 (desktop: 1.07).

::: {.nota}
**The sill at 1.66 times** — Without the shadow, the area under the door sills showed, between the edge of the car and the ground, the hangar floor (from the environment map) seen at infinity: light that in the real scene is blocked. With the shadow, the region drops to 1.08 times the path trace (1.07 on the desktop).
:::

## 7.10 Frame-time control and the still frame

The draw target has had four designs:

1. **Draw target.** It started at 260k drawn and was cut at once when the GPU median frame passed 16.5 ms. It rose 6 % a second while under 14 ms.
2. **~227k drawn.** For a still eye, the $\tau$ floor was bisected until about 227k were drawn (v6's 60 fps cut).
3. **$\tau$ floor from frame time, plus a still frame**, with a 16.5 ms target.
4. **Quality first**: image quality takes precedence over frame rate, with a 33 ms target. This is the current version.

::: {.nota}
**Frame-time control** — The number of drawn gaussians determines the GPU time almost linearly (§7.12.1). The four versions:

1. **Draw target**: a number of splats that dropped as soon as the median exceeded 16.5 ms (just below the 60 Hz period, 16.67 ms) and rose 6 % per second when there was headroom (<14 ms). A *fast down, slow up* law.
2. **~227k fixed** with the eye still: the number v6 drew at 60 fps.
3. **Frame-time $\tau$ floor**: controlling $\tau$ instead of the number of splats, because $\tau$ is what governs the cut and its quality is uniform across the image.
4. **Quality first (33 ms)**: accepting 30 fps in exchange for more detail.
:::

In the current version, the GPU time of a frame is measured from its start to `onSubmittedWorkDone`. Every 0.5 s, with $\bar t$ the median of at least 8 moving frames and $T = 33$ ms:

::: {.nota}
**`onSubmittedWorkDone`** — `GPUQueue.onSubmittedWorkDone()` returns a promise that resolves when the GPU has finished everything submitted up to that point. Measuring from the start of the frame (or from the end of the previous one, if the GPU was still busy) to that resolution gives an approximate GPU time without needing `timestamp-query`, which is an optional feature and not always available in Safari/iOS. It includes some CPU latency, but it is robust. The **median** of at least 8 frames discards isolated peaks (a frame with run decoding, a garbage collection).

*References:* [46] (WebGPU, `GPUQueue.onSubmittedWorkDone()`).
:::

$$
\tau_{\text{floor}} \leftarrow
\begin{cases}
\tau_{\text{floor}}\cdot\min\!\left(1.6,\ \sqrt{\bar t / (0.91\,T)}\right) & \bar t > T,\\[2pt]
\tau_{\text{floor}}/1.04 & \bar t < 0.85\,T \text{ for two periods},\\
\tau_{\text{floor}} & \text{otherwise},
\end{cases}
\qquad \tau_{\text{floor}} \in [1.5,\ 8]\ \text{px}.
$$

::: {.nota}
**The control law** — Every 0.5 s, with $\bar t$ the median and $T=33$ ms:

- **Too slow** ($\bar t>T$): $\tau$ is multiplied by $\sqrt{\bar t/(0.91T)}$, just enough (with a 9 % margin) for the time to land somewhat below the target, capped at ×1.6 per step so as not to overreact to an anomalous median.
- **Headroom** ($\bar t<0.85T$ for two periods in a row): $\tau$ drops by 4 % (÷1.04). Going down slowly and only after sustained headroom avoids oscillation.
- **Between $0.85T$ and $T$**: left alone (dead band, hysteresis).
- $\tau_{\text{floor}}\in[1.5, 8]$ px: below 1.5 the extra detail is not noticeable in motion; above 8 the car degrades too much.

It is an asymmetric multiplicative controller, related to TCP's AIMD: fast reaction to overload, slow recovery.

*References:* Chiu, D.-M., Jain, R. (1989), "Analysis of the Increase and Decrease Algorithms for Congestion Avoidance in Computer Networks" (AIMD).
:::

The square root follows from the number drawn scaling about as $1/\tau^2$.

::: {.nota}
**Why the square root** — A node's projected error is $\varepsilon p/d$; with a larger $\tau$ the cut stops at nodes with a larger error, and since a node's typical error scales with its size, the on-screen size of the drawn nodes grows roughly in proportion to $\tau$. The number of nodes covering a surface of fixed screen area scales as $1/\text{size}^2 \propto 1/\tau^2$. If time is proportional to the number drawn, $t\propto\tau^{-2}$, and to reduce the time by a factor $\bar t/T$, $\tau$ must be multiplied by $\sqrt{\bar t/T}$.
:::

**The still frame.** After 300 ms with no input and no camera change, the moving cut is pinned in the node pool, $\tau$ drops to 0.5 px, and the page limit rises from the moving pool (14 MB of traversal) to the still pool (28 MB). One fine cut is drawn whatever it costs, and drawn again as more pages land. Then nothing is redrawn until a touch, which brings the pinned moving cut back in that same frame and evicts down to the moving budget at 32 pages an update. The budget is about 100 MB of car data at rest and about 55 MB while moving. Measured on the emulated phone, close up and still: 529k drawn at $\tau$ 1.23 with 95.6 MB of car data. A second measurement gave 531k at $\tau$ 1.22 in 118 ms, at a simulated 190 ms a million, with the first frame after a touch at 39–47 ms.

::: {.nota}
**The still frame** — The idea: in motion fluidity matters, at rest quality matters. After 300 ms with no input and no camera change:

- the moving cut is **pinned** in the node pool (kept so that it can be returned to instantly);
- $\tau$ drops to 0.5 px (almost leaf-level detail) and the traversal limit rises from 14 to 28 MB;
- that fine cut is drawn once, however long it takes (118 ms for 531k: a one-off frame does not have to meet 33 ms), and redrawn as more pages arrive;
- after that nothing is redrawn (saving battery and heat) until a touch brings back the pinned moving cut and evicts down to the moving budget at a rate of 32 pages per update (evicting everything at once would cause a hitch).

**The figures** — Budget: ~100 MB at rest, ~55 MB in motion. Close up and still, on the emulated phone: 529k drawn with $\tau$ 1.23 and 95.6 MB of data. "At 190 ms per million simulated": the emulator adds to the measured time an artificial delay proportional to the number drawn (the page's `simMsPerM` parameter), to imitate the iPhone's GPU so that the control reacts as it would on the device. The first frame after a touch, 39–47 ms.
:::

## 7.11 Memory accounting

The 50–60 MB budget applies to the **car data** only. Car data is the resident pages and runs, the WASM traversal, the node worker's heap and buffers, and the GPU records. The **render state** is counted apart: sharp records, atlas, shadow, targets, sort. It carries the image quality, and an iPhone 15 tab held about 1.2 GB in the stress probes. The **page** itself (JS heap, three.js, domes) is a third figure.

::: {.nota}
**Three accounts** — (1) **Car data**: what grows with the asset and with the cut (resident pages and runs, the WASM traversal, the node worker's heap and buffers, GPU records): the 50–60 MB budget applies to this. (2) **Render state**: sharp-lobe records, atlas, shadow, targets, sort buffers; it depends on the resolution and the quality options, not on the asset. (3) **The page**: the JavaScript heap, three.js, the domes (textures of the visible sky). Separating them makes reasoning possible: shrinking the asset does not shrink the render state, and vice versa.

**1.2 GB in an iPhone 15 tab** — In the stress probes the tab reached that total without being killed, which gives an empirical ceiling for that model (not a guaranteed limit: it depends on the system's memory pressure).
:::

| state | data | car data | render state | page |
|---|---|---:|---:|---:|
| six parts, $\tau$ 1, emulated | CMPC v1 | — | GPU 53 MB total (peak 59.6): data 24.7, fx 9, target + dome 7.9, env 2.7, sort 4.6, canvas 3.8 | — |
| whole car, $\tau$ 3, first v4 streamer | v5 | ≈60 MB CPU + GPU (page pool at the same $\tau$: ≈87) | — | — |
| whole car, $\tau$ 3, class traversal records | v6 | Camera 55.1, close 59.0, far 49.1 (all together) | | |
| whole car, $\tau$ 3, after the atlas | v6 | Camera **39.0**, close **43.8**, far **33.0** | 24.1 / 26.0 / 21.7 | ~39 → 25 → ~17 (two reductions) |
| whole car, $\tau$ 3 | v7 | 31–36 | — | — |
| still frame, close up | v7 | **95.6** (529k, $\tau$ 1.23) | — | — |

::: {.nota}
**The table, row by row** —

- **Six parts, CMPC v1**: the first prototype; 53 MB of GPU in total (peak 59.6), broken down as: data 24.7, effects 9, target + dome 7.9, environment 2.7, sort 4.6, canvas 3.8.
- **Whole car, first v4 streamer (v5 data)**: ≈60 MB of CPU+GPU; the whole page pool at the same $\tau$ would have needed ≈87, hence v4.
- **Records by class (v6)**: 55.1 / 59.0 / 49.1 MB in the Camera / near / far views.
- **After the atlas (v6)**: 39.0 / 43.8 / 33.0 MB of car data. The drop (~16 MB) comes from freeing environment buffers and sizing records to what is drawn; the render state separately (24.1 / 26.0 / 21.7), and the page from ~39 to ~17 MB in two reductions.
- **v7**: 31–36 MB in motion with $\tau$ 3: comfortably within budget.
- **Close-up still frame (v7)**: 95.6 MB, the at-rest budget.
:::

Several reductions were each byte-identical in the image:

- the relight sky's floats rebuilt from RGBE on upload (1.6 MB held instead of 8 MB for a 1k dome);
- the visible dome in rg11b10ufloat on the phone;
- one shared occlusion codebook (4.3 MB of JS arrays before);
- sharp records sized to the drawn slots (−0.3 to −2.2 MB);
- the WASM shipped as a separate file, with the layout parsed in a worker.

::: {.nota}
**Byte-identical reductions** —

- **RGBE** (the Radiance `.hdr` format: three 8-bit mantissas and a shared 8-bit exponent, 4 bytes per texel) is decoded to floats only when the sky is uploaded to the GPU; the floats (16 B/texel) are not retained: 1.6 MB instead of 8 MB for a 1k dome.
- **`rg11b10ufloat`**: a 32-bit-per-texel texture format (two 11-bit float channels and one 10-bit, with no sign or alpha), half of `rgba16float`, for the *visible* dome (the background), which needs no more precision.
- **A shared occlusion codebook** in a `Uint32Array`: ordinary JS arrays store each number as an 8-byte value plus headers; 4.3 MB become a fraction of that.
- **Sharp-lobe records sized to what is drawn**, not to the maximum.
- **The WASM as a separate file**: embedded in the JS bundle (in base64) it takes 33 % more and sits in memory twice (the string and the compiled module); as a file it is compiled while streaming (`WebAssembly.instantiateStreaming`).

*References:* Ward, G. (1991), "Real Pixels", *Graphics Gems II* (RGBE); [46] (formats `rg11b10ufloat`, `rgb9e5ufloat`, `rgba16float`).
:::

The earlier per-gaussian accounting explains why the compact path was needed. The previous relight kept 308 B a gaussian plus 29 MiB fixed: 147 / 206 / 323 MiB at 0.4 / 0.6 / 1.0M. The compact path keeps 42.4 B + 16 B per drawn splat + about 3 MiB.

::: {.nota}
**The check** — $308\ \text{B}\times N + 29$ MiB: with $N=0.4$M, $117.5+29\approx147$ MiB; with 0.6M, $176.2+29\approx206$; with 1.0M, $293.7+29\approx323$. The compact path: $42.4\ \text{B}\times N + 16\ \text{B}\times D + 3$ MiB, about 5–6 times less. Without this change, 1M gaussians did not fit on an iPhone.
:::

## 7.12 Measured results

### 7.12.1 The real iPhone

**Stress page, 10–11 October** (synthetic ramps over real Corvette splats; medians):

::: {.nota}
**The stress page** — A page that draws synthetic ramps of 200k to ~1M real Corvette gaussians, with each render path, measuring the median fps on the device. "Synthetic" because the number drawn is set by hand (it does not come from a cut), to isolate the per-splat cost.
:::

| path | 200k | 400k | 600k | 800k | ~1M | notes |
|---|---:|---:|---:|---:|---:|---|
| plain 3DGS, fps | 58.8 | 58.8 | 45.5 | 34.5 | 26.3–27 | cool phone; ~37 ms a million |
| transfer, earlier "by" path, fps | 38.5 | 12.3 | 7.2 | 5.2 | 3.7 | 362–469 MB GPU |
| compact, synthetic basis, ms (fps) | 30 (33.3) | 63 (15.9) | 101 (9.9) | 137 (7.3) | 154 (6.5) | 20.8–79.8 MB GPU |
| compact, basis fitted on car data, optimised shade, ms (fps) | 27 (37) | 47 (21.3) | 70 (14.3) | 95 (10.5) | 108 (9.3) | 35 MB GPU at 400k |

::: {.nota}
**The rows** —

- **Plain 3DGS**: Spark's draw without relight (fixed SH colour); ~37 ms per million: the minimum sort + rasterisation cost on that iPhone.
- **Transfer, earlier "by" path**: the desktop relight (expanding the transfer); at 400k it already drops to 12 fps and uses 362–469 MB of GPU: not viable.
- **Compact, synthetic basis**: the compact path with a test basis (not fitted); 20.8–79.8 MB of GPU.
- **Compact, fitted basis and optimised shade**: the final version; 108 ms for ~1M (9.3 fps) and 35 MB of GPU at 400k.

The difference between plain 3DGS (37 ms/M) and compact (~115 ms/M) is the price of the relight: about 3 times the cost of the pure draw.
:::

Model for the optimised shade: about 115 ms a million drawn + 4 ms, so 30 fps holds about 250k drawn and 60 fps about 110k. The first second of each step ran 2–3× faster than the sustained rate (clock or thermal state), and a run following four runs in 20 minutes was half as fast from 600k up. Sustained figures are used for planning.

::: {.nota}
**The linear model** — $t \approx 115\ \text{ms}\cdot D/10^6 + 4$ ms. For 30 fps ($t\le33.3$): $D\le(33.3-4)/115\times10^6\approx255$k; for 60 fps ($t\le16.7$): $D\approx110$k. This sets the scale of the cut that can be asked for in motion.

**Thermal state** — Phone SoCs lower their clock when they heat up (*thermal throttling*). The first second of each step ran 2–3 times faster than the sustained rate, and a run after 20 minutes of use was half as fast from 600k on. Only the sustained figures are valid for planning.
:::

**Whole-car runs on the device** (v7 data):

| run | view | drawn | $\tau$ | fps | observation |
|---|---|---:|---:|---:|---|
| first | default | 106k | 8.00 | 53 | fluid; the $\tau$ 8.00 was the old frame-time "breath" stuck at its ceiling (fixed) |
| first | close up | 160k | — | 32 | soft logo and grille |
| second | close up | — | — | — | still soft |

::: {.nota}
**The runs on the device** — The first default run (106k drawn, 53 fps) was smooth, but with $\tau$ stuck at 8.00, the ceiling of the old control (frame-time "breathing" that got stuck at the top), since fixed. Close up, 160k at 32 fps, with the logo and the grille soft.
:::

The estimated shade cost on that iPhone is about 190 ms a million. The softness close up comes from the cloud, not the player: the logo baked at 1280 px is blurred even with all its leaves, and crisp at 5120 px. The car5120 close-up level (v8) addresses it.

::: {.nota}
**190 ms per million** — More than the stress page's 115: the real car triggers more expensive paths (per-pixel sharp lobes, glass, more overdraw close up), and the thermal state varies. It is the figure the emulator's simulation uses (§7.10).

**The softness comes from the cloud** — With every leaf drawn, the logo baked at 1280 px is still blurred: the limit is the bake's resolution, not the cut or the compression. At 5120 px (car5120, v8 data) it comes out sharp. It is an example of why the paper separates the format's error from the master's error.
:::

### 7.12.2 The emulated iPhone

| data | view | drawn / resident | $\tau$ | download | fps |
|---|---|---|---:|---:|---:|
| six parts, paged | default | 174k / 384k | 1.55 | 25 MB | 61 |
| six parts, CMPC v1 | default | 199k / 555k | 1 | 24.6 MB | 60 |
| same, CMPC v2 + `.cmpa` | default | 199k / 555k | 1 | 17 MB | 60 |
| whole car, v3 (24 MB pool, fov = h) | default | 147k / 580k | 6.75 | 20 MB | 60 |
| whole car, v5 (CMPC v4) | default | 226k / 1016k | 3 | 36 MB read | 60 |
| whole car, v7 | default | 216k / 1318k | 2.22 | — | 60 |
| whole car, v7, later streaming revisions | default | 234k | 1.95 | — | 60 |
| whole car, v7, still frame | close up | 529k | 1.23 | — | (one frame) |

::: {.nota}
**The emulated iPhone table** — "Drawn / resident" are nodes in the cut against nodes with loaded data; $\tau$ the threshold at which the control settled; "download" the MB transferred until settling; 60 fps is the vsync ceiling of the emulated screen.

- **Six parts**: from 25 MB to 17 MB of download when the tables move to the `.cmpa` (CMPC v2).
- **Whole car v3**: with a 24 MB pool, $\tau$ has to rise to 6.75 (147k drawn).
- **v5 (CMPC v4)**: the node pool allows $\tau$ 3 with 226k drawn and 1M resident in the traversal (36 MB read).
- **v7**: $\tau$ 2.22 and then 1.95 with the streaming improvements (234k).
- **Still frame**: 529k with $\tau$ 1.23, one frame.
:::

### 7.12.3 Against the path trace, by data version

`car-compact-tx` data versions are distinct from the `CMPC` format versions:

::: {.nota}
**Two separate versionings** — The `CMPC` versions (v1–v4) are of the tier-4 *format*; the `car-compact-tx` versions (v3–v8.2b) are published *data sets* of the car, which combine an athenea bake, a format version and conversion options.
:::

| set | content |
|---|---|
| v3 | occl8k, CMPC v3 |
| v4 | athenea's closed2 trees |
| v5 | CMPC v4 runs |
| v6 | athenea's mrough trees, per-material roughness, 1.35 GB |
| v7 | car1280, 1.81 GB |
| v8 | car5120: 24 trees, 74.5M leaves, 148.9M nodes, shared 8182-entry occlusion (§6.9) |
| v8.2b (final) | car5120 rebuilt (§6.2.1): 149.0M nodes, 8181-entry occlusion, a shared 512-cluster basis refitted with the 5120 data and four bases of 256 of the asset's own (Part_2, Infinite_Black, two parts of Part_3 and Front_Bumper_001), 83 files, 12.77 GB; transfer relMSE 0.0054 over the car |

::: {.nota}
**The data sets** —

- **v3**: an 8k-entry occlusion codebook and CMPC v3. **v4**: athenea's `closed2` trees (parts marked as closed, with cones for the back-cull). **v5**: CMPC v4 runs. **v6**: `mrough` trees (per-material roughness), 1.35 GB. **v7**: the 1280 px bake, 1.81 GB.
- **v8**: the 5120 px close-up: 24 trees, 74.5M leaves, 148.9M nodes (≈2 nodes per leaf, as expected in a binary merge tree), shared occlusion of 8182 entries.
- **v8.2b (final)**: car5120 rebuilt (§6.2.1), 149.0M nodes, 12.77 GB in 83 files. One shared 512-cluster basis refitted on the 5120 data, plus four 256-cluster bases of their own for parts whose transfer departs from the rest (Part_2, Infinite_Black, two parts of Part_3 and Front_Bumper_001): an atypical part would make the common basis worse. Transfer relMSE 0.0054.

*References:* §6.2.1 and §6.9 of the paper.
:::

Unless noted, the figures below are PSNR at athenea's camera at 1920 × 1080, by the validate-corvette harness's regions.

::: {.nota}
**validate-corvette** — The harness that renders the car in the player with athenea's camera and compares it with the path trace by region (whole car, door, sill, bumper, hood), giving PSNR per region.
:::

| version | car | door | sill | bumper | hood | source / condition |
|---|---:|---:|---:|---:|---:|---|
| v6 | 19.33 | 33.77 | 31.87 | 20.21 | 14.16 | compact path |
| v6, Toksvig unwiden | 19.68 → 20.58 | +1.0 | — | — | +1.3 | per-pixel detail on |
| v7 (first conversion) | +0.5 vs v6 | −2.3 | −2.7 | — | +0.9 | specks on folds |
| v7 + pixel spread, $\tau$ 1 | — | **35.53** | **32.36** | relMSE 0.199 | 14.46 | pixel spread on |
| v7, current defaults | 19.03 | 35.52 | 32.36 | 22.28 | 14.68 | degree-1 field |
| v8.1 | 19.22 | 35.92 | — | — | — | pool 3000, $\tau$ 1; draws 4.0 M |
| v8.2b | 19.27 | 35.53 | — | — | — | pool 3000, $\tau$ 1; draws 2.26 M; logo from a close camera 18.12 dB (v8.1 17.75) |

::: {.nota}
**How to read the table** — The absolute figures are low (19 dB for the car, 14 dB for the hood) because the comparison is against the mesh path trace: it includes all the bake error (gaussians against triangles, degree-3 transfer, 11° cells) and differences in sharp reflections, which penalise PSNR heavily. The hood, almost a mirror reflecting the environment, is the most demanding region. What one reads are the *differences* between rows:

- Toksvig unwidening raises the car from 19.68 to 20.58 dB and the hood by +1.3.
- The first v7 conversion worsens the door (−2.3) and the sill (−2.7) through the specks on the folds; the pixel spread fixes them (door 35.53, sill 32.36).
- The v7 defaults (degree-1 field) keep those figures.
- v8.1 and v8.2b: with the 3000 pool and $\tau$ 1, v8.2b's door loses 0.39 dB against v8.1 but draws almost half as much (2.26M against 4.0M), and the close-up logo improves (18.12 against 17.75 dB).

*References:* §1.6 and §8 of the paper.
:::

The car crop metric of §1.6 (900 × 420 crop) is a different region. On it, the per-pixel sharp lobe in the compact path took the car from 16.49 to 17.46 dB (17.41 with lobes from 4 px), which is the desktop path's level. On the phone, $\tau$ 3 against $\tau$ 1 differs by 40 dB, so the remaining defects (hairs, streaks, blotches) come from the cloud, not from the cut. Lossless steps were checked against the previous image: the sharp records in 5 words, 60–80 dB; binned shading, 84 dB (run-to-run noise 78 dB); the glass-only terms, 188 dB.

::: {.nota}
**Car crop** — Another metric (§1.6): a 900 × 420 crop around the car, which gives less weight to the background. With it, the per-pixel sharp lobes raise the car from 16.49 to 17.46 dB, the level of the desktop path: the phone matches the desktop in quality on that metric.

**$\tau$ 3 against $\tau$ 1: 40 dB** — The images at $\tau$ 3 and $\tau$ 1 are 40 dB apart (relMSE $10^{-4}$): practically the same. So the remaining defects (hairs, streaks, blotches) do not come from the cut (they do not improve with more detail) but from the cloud.

**Lossless steps checked** — Each optimisation that was meant to be exact was compared with the previous image: sharp lobes in 5 words, 60–80 dB (f16 rounding); the grouped shade, 84 dB, above the noise between two identical runs (78 dB: the order of the atomics on the GPU changes from one run to the next, and with it the order of some floating-point sums); the glass-only terms, 188 dB (equality except for stray bits).
:::

## 7.13 Limitations and open work

- **v8.2b**, the final data set (car5120 rebuilt, §6.2.1), is published: against the path trace it gains on the car, the grille, the chrome and the bumper and on the close logo, and loses 0.39 dB on the door against v8.1 while drawing about half as many Gaussians. Its download and frame times on the real phone are not measured.
- Real-device timing exists only for the stress ramps and the two whole-car runs. The emulated phone's speed is not the phone's.
- The front bumper's chrome strip is 38 % brighter than the path trace. The lamps have no emission in the bake. The door's open reflections (1.13) come from 16 × 16 cells of 11°.
- The compact traversal does not keep a merge's LoD size, so the combined size/error cut has nothing to guard with. The error cut alone did not beat the size cut on these data in every view.

::: {.nota}
**Limitations, explained** —

- **v8.2b not measured on the device**: the final set's download and frame times on a real iPhone have not been measured; the device figures are for v7.
- **Few device measurements**: only the ramps and two runs; emulated fps are not the phone's.
- **Bumper chrome +38 %, lamps without emission, door reflections 1.13**: errors of the master (the bake), not of the format. The 16 × 16 cells cover $\approx 11^\circ$ each (about $180^\circ/16$), so occlusion finer than that in the reflections is not represented.
- **No LoD size in the compact traversal**: the combined size/error cut needs the merge's LoD size (the `LODS` section) so as not to accept large merges with small error; the compact record does not store it. The error cut alone did not beat the size cut in every view.

*References:* §6 and §9 of the paper.
:::
