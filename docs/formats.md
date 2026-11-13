English · [Español](formats.es.md)

# athenea's file formats

This is the specification of the files athenea writes in formats of its own:
`.athc` (a cloud with its levels of detail, read chunk by chunk), `.athz` (a
paged, entropy-coded cloud, written by `athenea athz` on the `athz` branch)
and `.athv` (a video of gaussians, and the messages a stream of it
carries). It is written to be enough for an
independent reader. The code it describes is cited at each structure; where
the two disagree, the code is right and this file is wrong.

`docs/operations.md` says which command writes each file and with which
options (§2.4 `convert`, §8.2 Formats); `docs/decisions.md` says why each
field is what it is. The USD schema and its `athenea:splat:*` primvars are
specified in `docs/operations.md` §4.3, not here.

## 1. Conventions

- **Byte order.** Little endian throughout, except where a field says
  otherwise (the HEVC NAL unit lengths of `.athv`, which are big endian as
  HEVC's own `hvcC` framing has them).
- **Types.** `u8`, `u16`, `u32`, `u64` unsigned; `i32` two's complement;
  `f16` IEEE 754 binary16; `f32` binary32; `f64` binary64. A *word* is a
  `u32`. A *half pair* is a word whose low 16 bits are the first `f16` and
  whose high 16 bits are the second.
- **Arrays.** Blocks are laid out as structures of arrays: each attribute of
  every element of a block, then the next attribute.
- **Offsets** are in bytes from the start of the file unless a field says
  from where.
- **Refusal.** A reader refuses a file whose magic, version, counts or
  offsets do not add up, and a flag bit it does not know: a later bit may add
  an array, and a reader that skipped it would read every array after it
  from the wrong place.
- **Writing.** Every writer writes under a temporary name beside the
  destination (`<name>.partial`) and renames it when the file is complete, so
  a file under its own name is never half written.

## 2. A gaussian's words

`.athc` keeps the device's own packing (`shaders/athenea/common/packing.slang`,
`shaders/athenea/scene/splat_encoding.slang`, `GpuSplats` in
`modules/scene/include/athenea/scene/GpuClouds.h`); `.athz` decodes into it.
Each element of a cloud is the following words.

### 2.1 Position: `f32` × 4, 16 bytes

| Offset | Type | Field |
|---|---|---|
| 0 | f32 × 3 | x, y, z: the centre in the cloud's own space, in the units of its source (metres for every athenea writer) |
| 12 | f32 | opacity, linear, 0 to 1 (the sigmoid already applied) |

No up axis is recorded. A merged group's opacity is at most 0.99.

### 2.2 Shape: `u32` × 4, 16 bytes

| Word | Bits | Field |
|---|---|---|
| 0 | all | rotation, smallest three (below) |
| 1 | 0–15, 16–31 | `f16` ln(scale x), `f16` ln(scale y) |
| 2 | 0–15, 16–31 | `f16` ln(scale z), `f16` base colour r |
| 3 | 0–15, 16–31 | `f16` base colour g, `f16` base colour b |

**Rotation.** A unit quaternion (x, y, z, w). Bits 0–1 are the index of its
largest component by magnitude (0 x, 1 y, 2 z, 3 w); the quaternion is
negated before packing if that component is negative, so it is implied
positive. The other three, in order of index, are in bits 22–31, 12–21 and
2–11, each `round(1023 · (q / √½ · 0.5 + 0.5))` clamped to 0..1023. To
decode: each small component is `(v / 1023 · 2 − 1) · √½`, and the largest
is `√max(0, 1 − Σ small²)`. The rotation turns the gaussian's own axes
(columns of its rotation matrix) into the cloud's space; the scales are its
standard deviations along those axes.

**Scales.** Natural logarithms in `f16`, clamped to ±65000 before
conversion. A scale whose logarithm is below −100 is exactly zero: the
gaussian is a **surfel** (a 2D gaussian, a flat disc). athenea's writers put
the zero axis third, so the first two span the disc and the third is its
normal. A surfel's third `f16` v also carries its **hardness** p, the falloff
exp(−(|u|^p + |v|^p)/2) in its own axes: p = 2 + (−v − 1024) / 64 where
−60000 < v ≤ −1024 (writers keep p ≤ 32, v ≥ −2944), and p = 2, a gaussian,
otherwise (writers use −65000). Its first two scales are then the
moment-matched ones, so a reader that ignores p draws the gaussian with the
same moments.

**Base colour.** The DC term as a colour: `0.5 + 0.28209479 · f_dc` for a
3DGS spherical-harmonics DC coefficient `f_dc`. It may leave 0..1. It is
display-encoded (sRGB as a capture has it) unless the file says its colours
are linear light (`.athc` flag bit 1).

### 2.3 Harmonics: `shWords` words

The rest coefficients (degree 1 and above), `f16`, red, green and blue of
basis 0, then of basis 1, and so on, two to a word, the first in the low
half; an odd count leaves the last high half zero. The bases are in 3DGS's
order (degree 1: y, z, x; degree 2: xy, yz, 3z²−1, xz, x²−y²; degree 3 in
the same convention). `restPerColour` is 0, 3, 8 or 15 (degree 0 to 3), and
`shWords = (3 · restPerColour + 1) / 2`: 0, 5, 12 or 23. A writer may keep
one zero word at degree 0.

### 2.4 Optional words

Each is one array of the block, present where the file says.

| Array | Words | Encoding |
|---|---|---|
| normals | 1 | the shading normal, octahedral (below): x in bits 0–15, y in bits 16–31, each `round(65535 · (p · 0.5 + 0.5))` |
| emission | 1 | the radiance the gaussian emits, linear RGB9E5 (below) |
| pbr | 1 | the material (below) |
| lobes | 3 | the layers over the base (below) |
| transfer | `transferWords` | the transfer's `transferCount` values as half pairs (below) |
| open directions | `shadowWords` | one bit a cell of an octahedral grid (below) |
| curvature | 1 | the shape operator (below) |

**Octahedral map** (`shaders/athenea/common/octahedral.slang`). A direction
n is divided by |x| + |y| + |z|; where z < 0, (x, y) becomes
`(1 − |y|, 1 − |x|)` with the signs of x and y; the result in −1..1 maps to
0..1 as `p · 0.5 + 0.5`. Decoding: f = 2u − 1, n = (f.x, f.y,
1 − |f.x| − |f.y|), t = max(0, −n.z), then x and y each moved t towards zero,
and normalised. A zero normal packs as +z.

**RGB9E5.** Bits 0–8, 9–17, 18–26 the red, green and blue mantissas m, bits
27–31 the exponent e: each channel is `m · 2^(e − 24)`. Unsigned, up to
65408; negative and NaN are written as zero.

**pbr.** Bits 0–7 metallic, 8–15 roughness, 16–23 transmission, each
`round(255 · v)`. Bit 24: a thin wall (a sheet whose transmission is its own
transparency). Bit 25: a Schlick metal (OpenPBR's and glTF's conductor). Bit
26: bits 27–31 hold the material's own roughness, before the conversion
widened it by the spread of normals under the gaussian, as
`bits / 31`.

**Lobes.** Bytes, each `round(255 · v)` unless said:

| Word | Bits 0–7 | 8–15 | 16–23 | 24–31 |
|---|---|---|---|---|
| 0 | specular colour r | g | b | specular weight |
| 1 | coat weight | coat roughness | bits 0–6 coat IOR as `1 + b / 64`; bit 7 coat darkening | specular IOR as `1 + b / 128` |
| 2 | sheen colour r | g | b | sheen roughness |

A gaussian without lobes is specular weight 1, colour white, IOR 1.5, no
coat and no sheen.

**Transfer** (`shaders/athenea/common/transfer_layout.slang`). How the
gaussian answers light, path traced once. The count of values is the
layout:

| `transferCount` | Direct | Indirect | Field |
|---|---|---|---|
| 9 | 9 | — | — |
| 36 | 9 | 27 | — |
| 84 | 9 | 27 | 48 |
| 16 | 16 | — | — |
| 64 | 16 | 48 | — |
| 112 | 16 | 48 | 48 |
| 10 | zonal (below) | | |

The direct half is one value a harmonic basis (degree 2: 9, degree 3: 16),
in the standard real SH order. The indirect half is three a basis, red,
green, blue of basis 0 first. The reflected field is 16 rgb coefficients,
likewise. Values are `f16`; value 2k is the low half of word k.
`transferWords = (transferCount + 1) / 2`. A zonal transfer (10) is two
lobes, each its axis in the gaussian's own frame (u, v) and three zonal
coefficients (z0, z1, z2), in that order, with two words of open
directions in the same frame.

**Open directions.** Which ways out of the gaussian the bake found open,
one bit a cell of a side × side octahedral grid over the whole sphere: side
8 in 2 words (the first transfer), 16 in 8 and 32 in 32 (a TX transfer). Cell
(x, y) is bit `(y · side + x) & 31` of word `(y · side + x) >> 5`; its
direction is the octahedral decoding of `((x + 0.5) / side, (y + 0.5) /
side)`. A set bit is open.

**Curvature.** The shape operator (uu, uv, vv per unit of length) in the
gaussian's own axes. Bits 24–31 are `top`, an `i8`: ⌈log2⌉ of the largest
component. Bits 0–7, 8–15, 16–23 hold uu, uv, vv: bit 7 the sign, bits 0–6
q, the size `2^(top − 16 + q · 16 / 127)`; q = 0 is zero. A merged group of
more than one gaussian is flat (the word is zero).

## 3. `.athc`: a cloud with its levels of detail

Written by `athenea convert` (§2.4 of `docs/operations.md`) and
`writeAthc` (`modules/lod/src/Athc.cpp`); read by `readAthc`, and chunk by
chunk by `StreamingPool` (`modules/lod/include/athenea/lod/Athc.h`).

### 3.1 Layout

```
offset 0      header (104 bytes), then the extra header (32 bytes) where flag bit 4, 5 or 6 is set,
              then the source words (32 bytes) at 136 in every file
page 1        the level table, then the chunk table
aligned       the finest level's group starts
aligned       each level, coarsest first: one block
aligned       each chunk, in order: one block
```

Every block, and the start table, starts on a 4096-byte boundary, so a
chunk maps and faults in alone; the gaps are zero, and the file's length is
a multiple of 4096.

The splats are in Morton order (§3.4), cut into **chunks** of `chunkSplats`
consecutive splats (the last holds the rest). A **level** l holds one merged
gaussian (a **group**) for each octree cell at depth `level` that holds
splats.

### 3.2 Header

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 4 | char[4] | magic | `ATHC` |
| 4 | 4 | u32 | version | 1 or 2 |
| 8 | 4 | u32 | count | splats, > 0 |
| 12 | 4 | u32 | restPerColour | 0, 3, 8, 15 |
| 16 | 4 | u32 | shWords | harmonic words an element, 1 to 64 |
| 20 | 4 | u32 | levels | merged levels, ≥ 1 |
| 24 | 4 | u32 | chunkSplats | splats a chunk, > 0 |
| 28 | 4 | u32 | chunks | `⌈count / chunkSplats⌉` |
| 32 | 4 | u32 | finestGroups | groups of the finest level |
| 36 | 12 | f32 × 3 | boundsLo | the octree's lower corner |
| 48 | 4 | f32 | extent | the octree's edge: the bounds' largest edge |
| 52 | 12 | f32 × 3 | boundsMin | the splats' bounds |
| 64 | 12 | f32 × 3 | boundsMax | |
| 76 | 4 | u32 | flags | version 2; zero (padding) in version 1 |
| 80 | 8 | u64 | levelTable | offset of the level table: 4096 |
| 88 | 8 | u64 | chunkTable | offset of the chunk table |
| 96 | 8 | u64 | starts | offset of the start table |

A file is written as version 2 if and only if `flags` is not zero; a reader
of version 1 treats `flags` as zero.

**Flags.**

| Bit | Value | Meaning |
|---|---|---|
| 0 | 1 | every block carries normals |
| 1 | 2 | the colours are linear light (no array) |
| 2 | 4 | every block carries emission |
| 3 | 8 | reserved (proposal 009); refused |
| 4 | 16 | every block carries the material: pbr and, per the extra header, lobes |
| 5 | 32 | every block carries a transfer and, per the extra header, open directions |
| 6 | 64 | every block carries curvature |

Any other bit is refused.

**Extra header** (at offset 104, only where bit 4, 5 or 6 is set):

| Offset | Type | Field | Constraint |
|---|---|---|---|
| 104 | u32 | pbrWords | 1 if and only if bit 4 |
| 108 | u32 | lobesWords | 0 or 3 |
| 112 | u32 | transferCount | 9, 36, 84, 16, 64, 112 or 10 |
| 116 | u32 | transferWords | `(transferCount + 1) / 2`, ≤ 64; non-zero if and only if bit 5 |
| 120 | u32 | shadowWords | 0, 2, 8 or 32 |
| 124 | u32 | curvatureWords | 1 if and only if bit 6 |
| 128 | u32 × 2 | pad | zero |

**Source words** (at offset 136, in every file; zeros in one written before
them, which reads as "unsaid"): what the cloud's source said of itself that no
block carries. They are page 0's padding, so a reader that does not know them
skips them; no flag bit says they are there.

| Offset | Type | Field | Constraint |
|---|---|---|---|
| 136 | u32 | antialiased | 0 unsaid, 1 fitted without the Mip-Splatting filter, 2 with it (SPZ's flag); a reader refuses anything else |
| 140 | u32 × 7 | pad | zero |

### 3.3 Tables

**Level table** (`levels` entries, 16 bytes each, at `levelTable`), coarsest
first:

| Offset | Type | Field | Constraint |
|---|---|---|---|
| 0 | u32 | level | octree depth, 1 to 10; each entry one more than the one before |
| 4 | u32 | groups | > 0; the last entry's is `finestGroups` |
| 8 | u64 | offset | the level's block |

**Chunk table** (`chunks` entries, 16 bytes each, right after the level
table):

| Offset | Type | Field | Constraint |
|---|---|---|---|
| 0 | u64 | offset | the chunk's block |
| 8 | u32 | count | `min(chunkSplats, count − c · chunkSplats)` |
| 12 | u32 | pad | zero |

**Start table** (`finestGroups` words at `starts`): for each group of the
finest level, the index of its first splat in the Morton order. A group's
splats are the run up to the next group's start.

### 3.4 Blocks

A block of n elements is these arrays, each starting where the one before
ends:

| Array | Bytes | Present |
|---|---|---|
| positions | 16 n | always (§2.1) |
| shape | 16 n | always (§2.2) |
| harmonics | 4 · shWords · n | always (§2.3) |
| tail | 4 n | always: in a level's block, each group's cell code; in a chunk's block, each splat's finest-level group |
| normals | 4 n | flag bit 0 |
| emission | 4 n | flag bit 2 |
| pbr | 4 n | `pbrWords` = 1 |
| lobes | 12 n | `lobesWords` = 3 |
| transfer | 4 · transferWords · n | `transferWords` > 0 |
| open directions | 4 · shadowWords · n | `shadowWords` > 0 |
| curvature | 4 n | `curvatureWords` = 1 |

An element's bytes are therefore `16 + 16 + 4·shWords + 4 + 4·(normals) +
4·(emission) + 4·(pbrWords + lobesWords + transferWords + shadowWords +
curvatureWords)`, and a reader checks that each block fits in the file.

**Morton order and cell codes** (`shaders/athenea/common/morton.slang`). A
point's unit position is `(p − boundsLo) / extent`, clamped to 0..1; each
axis is quantised to 10 bits, `min(⌊1024 · u⌋, 1023)`, and the three are
interleaved into a 30-bit code with x in the highest bit of each triple
(bit 29 is x's top bit, 28 y's, 27 z's). The octree cell at depth r that
holds a point is its code's top 3r bits, `code >> (30 − 3r)`; that is a level
block's tail word. Splats are sorted by a key whose low part is their
centre's code and whose high part is a class a merge may not mix
(`shaders/athenea/lod/lod_class.slang`: 0–5 the axis and sign the shading
normal leans along most, 6 glass and sheets, which the cut never draws
merged). A group is one class in one cell, and its splats are one run of
the order. The class is not stored: a reader takes the groups from the
level blocks' tails, each splat's finest group from the chunk blocks'
tails, and the runs from the start table. Two groups of a level may share a
cell code.

**Merged groups** (`shaders/athenea/lod/lod_common.slang`). A group is one
gaussian with the moments of the splats under it, each weighed by
w = opacity × the area of its two widest axes: mean = Σ w μ / W, covariance =
Σ w (Σ + (μ − m)(μ − m)ᵀ) / W (eigen-decomposed into the rotation and
scales), colour = Σ w c / W, opacity = min(W / its own two-axis area,
0.99). A reader needs none of this to draw a file; it says what a level's
elements are.

**Drawing a cut.** A group is drawn when its cell's projected size is at
most the threshold and its parent's is not; a splat is drawn when the
finest level's cell holding it is not. Along any path down the tree the
test changes once, so every place is drawn at exactly one level. Threshold 0
draws every splat and no group.

### 3.5 Versions

- **Version 1**: no flags (the word is padding).
- **Version 2**: `flags` and, with bits 4–6, the extra header.
- **Source words** (13 Nov 2026, no version change): the antialiasing at
  offset 136. Files written before hold zeros there and read as unsaid.
- **`.lrtc`** (lucabRTrender, the engine's earlier name): version 1 of this
  layout under the magic `LRTC`. `athenea` reads it as a `.athc` and
  `migrateLrtc` rewrites the header alone.
- **Version 3** (`ATH3`) is not athenea's: sparkwebGPU writes it for the web
  (§6).

## 4. `.athz`: a paged, entropy-coded cloud

A `.athz` file holds a cloud with its levels of detail: each merged level's
groups and the cloud's own splats. Decoded, every element is the record a
`.athc` chunk block holds (§3, §2), bit for bit, followed where present by
curvature and by a skinning rig; only the transfer coded as CPCA (§4.12) is lossy. The elements are coded in pages, each page in up to four
independent layers, each layer as byte planes coded raw (bit-packed) or by
32-lane interleaved rANS, after a per-word integer transform.

Sources: `modules/lod/include/athenea/lod/Athz.h`, `modules/lod/src/Athz.cpp`,
`shaders/athenea/lod/athz_common.slang`, `athz_decode.slangh`,
`athz_encode.slang`, `athz_tables.slang`, `athz_cpca.slang`,
`athz_interleave.slang` (branch `athz`). The shaders `lod_pages`,
`transfer_codec`, `geometry_quantize`, `shadow_vq` and `splat_compact` are not
used by `.athz`; `athz_cpca.slang` reimplements the CPCA codec of
`transfer_codec.slang`.

### 4.1 Conventions

- All integers are little-endian, unsigned unless stated. `f16` is IEEE 754
  binary16, `f32` binary32, `f64` binary64, all little-endian.
- Multi-byte fields in the asset table are packed with no alignment padding
  (a `u64` may sit at an offset that is not a multiple of 8).
- A "word" is a `u32`. Byte `i` of a word (i = 0..3) is bits `8i .. 8i+7`,
  i.e. the byte at address `4w + i` of a little-endian word array.
- A "half" pair in a word: the low half is bits 0..15, the high half bits
  16..31.
- `aligned(x, a) = ceil(x / a) * a`.
- CRC-32 is zlib's `crc32`: the IEEE 802.3 polynomial 0x04C11DB7 (reflected
  form 0xEDB88320), initial value 0xFFFFFFFF, final XOR 0xFFFFFFFF, i.e.
  `crc32(0, bytes, n)`. A CRC over several byte ranges is the CRC of their
  concatenation.
- `zigzag(d) = (d << 1) XOR (0 - (d >> 31))` on 32 bits;
  `unzigzag(z) = (z >> 1) XOR (0 - (z & 1))`.
  `zigzag16`/`unzigzag16` are the same on 16 bits, results masked to 16 bits.
- `sub16x2(a, b)` / `add16x2(a, b)`: subtraction / addition of each 16-bit
  half independently, modulo 2^16.
  `zigzag16x2` / `unzigzag16x2`: `zigzag16` / `unzigzag16` on each half.

### 4.2 File layout

Source: `Athz.cpp` (`writeAthz`, `prefixPages`).

| Region | Offset | Size | Alignment |
|---|---|---|---|
| Header | 0 | 64 | — |
| Extensions (TLV) | `tlvOffset` (64) | up to 4032 | 8 per record |
| Zero padding | | to 4096 | |
| Asset table | `assetOffset` (4096) | `assetBytes`, a multiple of 64 | 4096 |
| Page table | `pageTableOffset` = `aligned(4096 + assetBytes, 64)` | `pages * 64` | 64 |
| Finest level's starts | `startsOffset` | `finestGroups * 4` | 4096 |
| CPCA basis (if CPCA) | `basisOffset` | `basisBytes` | 4096 |
| Layer K frames (if layer K) | `temporalOffset` | `temporalBytes` | 4096 |
| Section G | `sectionOffset[0]` | `sectionBytes[0]` | 4096 |
| Section A (if present) | `sectionOffset[1]` | `sectionBytes[1]` | 4096 |
| Section T (if present) | `sectionOffset[2]` | `sectionBytes[2]` | 4096 |
| Section K (if present) | `sectionOffset[3]` | `sectionBytes[3]` | 4096 |

Every gap is zero-filled, and the file is zero-padded to a multiple of 4096.
The writer lays the regions out in the order above, each later region at
`aligned(end of previous, 4096)`. A reader takes every offset from the header
and asset table and must not assume the writer's placement beyond what is
stated as checked below.

### 4.3 Header

Source: `Athz.cpp` (`encodeHeader`, `decodeHeader`).

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 4 | char[4] | magic | `'A' 'T' 'H' 'Z'` |
| 4 | 2 | u16 | version | 1 to 4; the writer writes 4 |
| 6 | 2 | u16 | headerBytes | 64 |
| 8 | 4 | u32 | flags | bit set, below |
| 12 | 4 | u32 | layerMask | bit L set when layer L is in the file (1 G, 2 A, 4 T, 8 K) |
| 16 | 8 | u64 | gaussians | the cloud's own splats; equals the asset table's `count` |
| 24 | 4 | u32 | pages | entries in the page table (fewer than the asset's total in a prefix) |
| 28 | 2 | u16 | runGaussians | R, elements a run: 32 to 256, a multiple of 32 |
| 30 | 2 | u16 | pageRuns | runs a page, at least 1 |
| 32 | 8 | u64 | assetOffset | the asset table's offset; the writer writes 4096 |
| 40 | 4 | u32 | assetBytes | the asset table's size, padding included |
| 44 | 8 | u64 | pageTableOffset | the page table's offset |
| 52 | 2 | u16 | entryBytes | 64 |
| 54 | 2 | u16 | reserved | 0 |
| 56 | 4 | u32 | tlvOffset | the first extension's offset (64), 0 for none |
| 60 | 4 | u32 | headerCrc | CRC-32 of bytes 0 to 59 |

Flags:

| Bit | Value | Name | Meaning |
|---|---|---|---|
| 0 | 1 | Transfer | layer T is present |
| 1 | 2 | Harmonics | `restPerColour > 0` (harmonics past the DC in layer A) |
| 2 | 4 | Temporal | layer K is present |
| 3 | 8 | Linear | colours are in linear light (the decoded cloud's `linear`) |
| 4 | 16 | Predicted | at least one plane entry (either kind) has a transform ≥ 4 (coded against the parent) |
| 5 | 32 | Emission | elements carry an RGB9E5 emission word |

Only bit 3 changes how the decoded cloud is interpreted. Bits 0, 1, 2, 4 and
5 are descriptive; the decoder takes the element layout from the asset table
and the transforms from the plane entries. A flag bit outside 0x3F is
refused.

### 4.4 Extensions

Source: `Athz.cpp` (`writeAthz`, `parse`).

A sequence of records starting at `tlvOffset`, each:

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 2 | u16 | type | 0 ends the list; 1 is the source name |
| 2 | 2 | u16 | flags | written 0; not read |
| 4 | 4 | u32 | length | payload bytes |
| 8 | length | u8[] | payload | |
| 8 + length | | | padding | zeros to `aligned(length, 8)` |

The next record starts at `8 + aligned(length, 8)` from this one. The list ends
with a type-0 record (8 zero bytes; its length is not used). A record that
runs past `assetOffset` is refused; walking stops at `assetOffset` if no
terminator is met. Unknown types are skipped.

- Type 1 (source): the name of the file the asset was converted from (the
  writer stores the input's file name, at most 2048 bytes, no terminator). Its
  character encoding is whatever the file system name was; the code treats it
  as bytes.

### 4.5 Asset table

Source: `Athz.cpp` (`encodeAsset`, `decodeAsset`).

At `assetOffset`, `assetBytes` long. Offsets below are for version 4; see
§4.15 for 1 to 3.

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 4 | u32 | bytes | the table's size, padding included; must equal the header's `assetBytes` |
| 4 | 4 | u32 | kinds | 2 (kind 0 merged groups, kind 1 the cloud's splats) |
| 8 | 4 | u32 | planes | P, byte planes an element: 4 × W (W, the coded element's words) |
| 12 | 4 | u32 | tableLog | n, log2 of a rANS table's size: 9 to 12 |
| 16 | 4 | u32 | tables | T, rANS tables in the file |
| 20 | 4 | u32 | levels | merged levels: 1 to 30 |
| 24 | 4 | u32 | count | the cloud's splats (> 0) |
| 28 | 4 | u32 | finestGroups | groups of the finest merged level |
| 32 | 4 | u32 | chunkSplats | elements a page: must equal R × pageRuns, at most 65535 |
| 36 | 4 | u32 | restPerColour | harmonic coefficients past the DC a colour: 0, 3, 8, 15 |
| 40 | 4 | u32 | shWords | harmonic words an element: 1 to 64 (one dummy word at degree 0) |
| 44 | 4 | u32 | normals | 1: a normal word an element |
| 48 | 4 | u32 | emission | 1: an RGB9E5 emission word an element |
| 52 | 4 | u32 | pbrWords | 0 or 1 |
| 56 | 4 | u32 | lobesWords | 0 or 3 |
| 60 | 4 | u32 | transferCount | values of the decoded transfer (112 for a degree-3 TX transfer) |
| 64 | 4 | u32 | transferWords | words of the decoded transfer an element (f16 pairs) |
| 68 | 4 | u32 | shadowWords | open-direction words an element |
| 72 | 4 | u32 | curvatureWords | 0 or 3 (v3+) |
| 76 | 4 | u32 | codedTransferWords | 0: transfer lossless; 4: transfer coded as CPCA (v3+) |
| 80 | 4 | u32 | cpcaClusters | C, 1 to 256 where CPCA (the writer writes 256) (v3+) |
| 84 | 4 | u32 | cpcaComponents | K, axes a block: 8 where CPCA (v3+) |
| 88 | 4 | f32 | cpcaLevels | quantisation steps a side (the writer writes 7.0) (v3+) |
| 92 | 8 | u64 | basisOffset | the CPCA basis's offset (v3+) |
| 100 | 8 | u64 | basisBytes | its size (v3+) |
| 108 | 4 | u32 | influenceWords | 0, or 8 with layer K (v4) |
| 112 | 4 | u32 | gradientWords | 0, or 3 with layer K and gradients (v4) |
| 116 | 4 | u32 | joints | J, the rig's joints (v4) |
| 120 | 4 | u32 | frames | F, authored frames; nonzero exactly when influenceWords is (v4) |
| 124 | 8 | u64 | temporalOffset | layer K's frames block offset (v4) |
| 132 | 8 | u64 | temporalBytes | its size: 64 + 8F + 24FJ when F ≠ 0 (v4) |
| 140 | 12 | f32[3] | boundsLo | the octree root's lower corner |
| 152 | 4 | f32 | extent | the octree root's edge (the bounds' largest edge) |
| 156 | 12 | f32[3] | boundsMin | the cloud's axis-aligned bounds, minimum |
| 168 | 12 | f32[3] | boundsMax | maximum |
| 180 | 8 | u64 | assetId | FNV-1a 64 of this table with this field zero (below) |
| 188 | 32 | u64[4] | sectionOffset | per layer G, A, T, K: the section's offset; 0 for an absent layer |
| 220 | 32 | u64[4] | sectionBytes | per layer: the section's size |
| 252 | 8 | u64 | startsOffset | the finest level's starts |
| 260 | 8 | u64 | startsBytes | must equal finestGroups × 4 |
| 268 | 16 × levels | LevelEntry[] | levels | coarsest first |
| … | 16 × 2P | PlaneEntry[2][P] | planeEntries | kind 0's P entries, then kind 1's |
| … | 512 × T | u16[T][256] | frequencies | each rANS table's 256 symbol frequencies |
| … | | | padding | zeros to a multiple of 64 |

`assetId`: FNV-1a 64-bit (offset basis 0xCBF29CE484222325, prime
0x100000001B3, applied byte by byte) over the whole encoded table, padding
included, with the `assetId` field set to 0 and every other field at its final
value. A prefix (§4.14) keeps the id of the file it was cut from without
recomputing it. No reader in the code checks it.

The coded element's word count W is

    W = 4 + 4 + shWords + 1 + normals + emission + pbrWords + lobesWords
        + transferWordsCoded + shadowWords + curvatureWords
        + influenceWords + gradientWords

where `transferWordsCoded` is `codedTransferWords` (4) under CPCA and
`transferWords` otherwise. `planes` must equal 4W.

#### LevelEntry (16 bytes)

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 4 | u32 | level | octree level: cells of edge `extent / 2^level` |
| 4 | 4 | u32 | groups | the level's merged groups (> 0) |
| 8 | 4 | u32 | firstPage | its first page in the page table |
| 12 | 4 | u32 | pages | `ceil(groups / chunkSplats)` |

Checked: entry 0 has `firstPage` 0; each entry's `firstPage` is the previous
one's `firstPage + pages`; each `level` is the previous one's + 1; the last
entry's `groups` equals `finestGroups`.

#### PlaneEntry (16 bytes, four u32)

Source: `athz_common.slang` (`athzEncoding` … `athzTransform`).

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 4 | u32 | coding | bits 0–7 encoding (0 raw, 1 rANS); 8–15 width (0 to 8, raw bits a symbol); 16–23 base (0 to 255); 24–31 layer (0 G, 1 A, 2 T, 3 K) |
| 4 | 4 | u32 | table | encoding 1: the rANS table index (< T); encoding 0: 0xFFFFFFFF |
| 8 | 4 | u32 | rawStart | encoding 0: where the plane's raw words start in a run's raw part, in words; encoding 1: 0 |
| 12 | 4 | u32 | transform | v2+: bits 0–7 the word's transform (0 to 8; the tail word's 0 to 3). v1: an encoder estimate, read as transform 0 |

Plane `b` (0 ≤ b < P) is byte `b & 3` of the element's word `b >> 2`. All four
planes of a word carry the same transform; the decoder reads it from the
word's last plane (`b & 3 == 3`). `width` and `base` are present on rANS
planes too but are only used for raw planes.

Table indices are assigned in order: kind 0's planes b = 0..P−1, then kind 1's,
each rANS plane taking the next index. Each kind therefore has its own table
per rANS plane.

`rawStart`, per kind and per layer: a running sum over the raw planes of that
layer in increasing `b`, each adding `ceil(R × width / 32)` words (0 for width
0).

#### Frequencies

Source: `athz_encode.slang` (`athzNormalize`), `athz_tables.slang`.

Table t is 256 u16 frequencies `f[0..255]` at
`frequencies + 512t`. They sum to exactly `2^n`; a table that does not is
refused. A symbol with frequency 0 never occurs. The decoder's slot table is
derived: with `cum[s] = f[0] + … + f[s−1]`, slot `x` in `[0, 2^n)` maps to the
unique symbol s with `cum[s] ≤ x < cum[s] + f[s]`. The writer only uses rANS
for planes with at least two distinct symbols, so every `f[s] < 2^n`; the
engine packs a slot as `s | f << 8 | cum << 20` (12 bits each for f and cum).

Normalisation used by the writer (informative; a reader needs only the
frequencies): over all elements of the kind, count `c[s]` of each residual
byte value; `f[s] = 0` if `c[s] = 0`, else `max(1, floor(c[s] × 2^n / total))`.
If the sum is below `2^n`, the deficit is added to the most frequent symbol
(the lowest-valued among equal counts). While the sum exceeds `2^n`, the
symbol with the largest `f` (lowest-valued among ties) loses
`min(f − 1, sum − 2^n)`. A plane is coded by rANS when it has at least two
symbols and `Σ c[s] (n − log2 f[s]) / 8 < 0.97 × total × width / 8`; otherwise
raw, with `base` the least symbol and `width` the bit length of
`max − min` (0 when one symbol occurs).

### 4.6 The element

Source: `athz_common.slang` (field constants, `athzField`), `Athz.cpp`
(`fieldWords`, `layerOfByte`, `formatOfCloud`).

An element is W words: the arrays below in this order, each taking its count
of words (0 where absent).

| Field | Words | Content | Layer |
|---|---|---|---|
| 0 positions | 4 | `.athc` position words (§3, §2) | G |
| 1 shape | 4 | `.athc` shape words (§3, §2) | words 0, 1 and bytes 0–1 of word 2: G; bytes 2–3 of word 2 and word 3 (the base colour): A |
| 2 sh | shWords | `.athc` harmonic words | A |
| 3 tail | 1 | merged group: its cell code at its level; splat: its finest merged group's index | G |
| 4 normal | normals | `.athc` normal word | G |
| 5 emission | emission | RGB9E5 | A |
| 6 pbr | pbrWords | material word | G |
| 7 lobes | lobesWords | material layers | G |
| 8 transfer | transferWords, or 4 under CPCA | f16 pairs, or the CPCA code (§4.12) | T |
| 9 shadow | shadowWords | open-direction bits | T |
| 10 curvature | curvatureWords | 3 f32: shape operator uu, uv, vv | G |
| 11 influences | influenceWords | 4 × (f32 joint index, f32 weight) | K |
| 12 gradients | gradientWords | 3 words, each two f16: low d w_k/du, high d w_k/dv, k = 0..2 | K |

The tail word's index is `tailWord = 8 + shWords`. The merged groups' cell
codes within a level are in ascending order, and a cell's parent at the level
above has code `cell >> 3`. Merged groups carry zeros in fields 11 and 12
(nothing skins them).

`layerMask` is the OR of the layers of all bytes of the coded element: G is
always present; A is present for every element layout (shape word 3); T where
fields 8 or 9 exist; K where fields 11 or 12 exist.

The decoded record (what a reader produces) is the coded element with field 8
replaced by the decoded transfer of `transferWords` words where CPCA is used.

### 4.7 Bands and pages

Source: `Athz.cpp` (`writeAthz`, `parse`), `athz_encode.slang` (`athzPageInfo`).

A band is one merged level (bands 0 to `levels − 1`, coarsest first) or the
cloud's splats (band `levels`). Each band's elements are cut into pages of
PG = `chunkSplats` = R × pageRuns elements; the last page of a band holds the
remainder. Page indices are global and in band order:

- level band `l` holds pages `levels[l].firstPage .. + levels[l].pages − 1`,
  page k of it holding the level's groups `[k·PG, min((k+1)·PG, groups))`;
- the splat band starts at `leafFirstPage = levels[last].firstPage +
  levels[last].pages` and has `leafPages = ceil(count / PG)` pages, page k
  holding splats `[k·PG, min((k+1)·PG, count))` in store (Morton) order.

The total is `leafFirstPage + leafPages`; the header's `pages` may be less
(§4.14), never more.

A page is `pageRuns` runs of R elements: run r holds the page's elements
`[r·R, r·R + here)`, `here = clamp(count − r·R, 0, R)`. Every page has all
`pageRuns` runs in every present layer, including runs with `here = 0`.

#### Page entry (64 bytes)

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 4 | u32 | info | bits 0–15 count (elements); 16–23 band; 24–31 octree level (0xFF for a splat page) |
| 4 | 4 | u32 | parentPage | merged page: the page of the level above holding the parent group of its first element; splat page: the finest level's page holding its first splat's group; 0xFFFFFFFF for band 0 |
| 8 | 4 | u32 | box0 | f16 lo.x (low half), f16 lo.y (high half) |
| 12 | 4 | u32 | box1 | f16 lo.z, f16 hi.x |
| 16 | 4 | u32 | box2 | f16 hi.y, f16 hi.z |
| 20 | 4 | u32 | error | f16 eps (low half), 0 (high half) |
| 24 | 16 | u32[4] | offset | per layer G, A, T, K: the page's bytes' offset within that layer's section |
| 40 | 16 | u32[4] | bytes | per layer: the page's bytes in that section; 0 for an absent layer |
| 56 | 4 | u32 | crc | CRC-32 of the page's bytes of every present layer, G, A, T, K order, concatenated |
| 60 | 4 | u32 | reserved | 0 |

- `count` must equal the band's element count for that page and `band` the
  band the page index falls in.
- The box: the minimum and maximum of the page's elements' position words 0
  to 2 (as f32), each converted to f16 and then moved one f16 step outward
  (down for lo, up for hi) where the conversion went inward, so the box
  contains every position.
- `eps`: `extent / 2^min(level, 30)` as f16 rounded up; 0 for splat pages.
- Parent page of a merged page in band l > 0: with `cell = tail(first element)
  >> 3`, `a` = the first index in level l−1 whose cell code is ≥ `cell`
  (clamped to `groups − 1`); `parentPage = levels[l−1].firstPage + a / PG`. Of
  a splat page: `levels[last].firstPage + min(tail(first splat),
  finestGroups − 1) / PG`.
- The box, `eps` and `parentPage` are not read by the engine.

Pages are stored contiguously in page order in each section: page i's bytes in
section L start where page i−1's end. For a present layer, every page's
`bytes[L] ≥ 4 × pageRuns` and `offset[L] + bytes[L] ≤ sectionBytes[L]`.

### 4.8 A page's layer

Source: `athz_encode.slang` (`athzEncodeRuns`, `athzPlace`),
`athz_decode.slangh`.

The bytes of layer L of one page:

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 4 × pageRuns | u32[pageRuns] | runEnd | runEnd[r]: the end of run r, in bytes from the start of the page's layer bytes |
| 4 × pageRuns | | | runs | run 0 starts at `4 × pageRuns`, run r > 0 at `runEnd[r−1]` |

`runEnd[pageRuns − 1]` equals the page's `bytes[L]`.

One run of layer L, for the page's kind k (kind 0 for level bands, 1 for the
splat band). `hasRans(k, L)` is true when any plane entry of kind k with layer
L has encoding 1; it is a property of the asset, so every run of that kind and
layer has the same structure.

| Present when | Size | Type | Field | Meaning |
|---|---|---|---|---|
| hasRans | 4 | u32 | words | N, 16-bit words in the stream |
| hasRans | 128 | u32[32] | states | each lane's final encoder state, the decoder's initial state |
| hasRans | 4 × ceil(N/2) | u16[N] + pad | stream | the 16-bit words; word j at byte 2j; a zero u16 pads an odd N |
| always | 4 × rawWords(k, L) | u32[] | raw | the raw planes of layer L |

`rawWords(k, L)` = Σ over raw planes of kind k in layer L of
`ceil(R × width / 32)` — always computed with the full R, even when the run
holds fewer elements. A plane's raw block is at word `rawStart` of the raw
part.

#### Lanes and symbol order

`steps = R / 32`. In every plane, element g of the run (0 ≤ g < R) belongs to
lane `g / steps` as its symbol `k = g % steps`: each lane owns a block of
`steps` consecutive elements. Elements with `g ≥ here` carry no symbol (no
rANS step; raw bits 0).

#### Plane coding order

Planes are coded and decoded in this order of i = 0..P−1:

    planeAt(i) = 4·tailWord + i                               if i < 4
                 4·w' + (i & 3), w' = (i−4)/4 + (1 if (i−4)/4 ≥ tailWord)   otherwise

i.e. the tail word's four bytes first, then every other word in element order,
bytes 0 to 3 of each. The four planes of a word are always consecutive.

#### Raw planes

For element g of the run, with the plane's `width` w > 0, the w bits at bit
position `g·w` of the plane's block (bit j of the block is bit `j % 32` of
block word `j / 32`; a value may straddle two words, low part first) are
`v`; the byte is `(v + base) & 0xFF`. With `width = 0`, every byte is `base`.

#### rANS

Source: `athz_common.slang` (header comment), `athz_decode.slangh`,
`athz_encode.slang` (`athzEncodeRuns`).

Parameters: state `x` is 32 bits, in `[2^16, 2^32)` between symbols; lower
bound `Lo = 2^16`; renormalisation by 16-bit words; frequency precision
`n = tableLog` (9 to 12). Each run and layer has one stream shared by 32 lanes.

Decoding a run's layer L (all arithmetic unsigned 32-bit):

    x[0..31] = states; ptr = 0
    for i in 0 .. P−1:
        b = planeAt(i); if layer(b) != L or encoding(b) != 1: continue
        f[], cum[] = table(b)
        for k in 0 .. steps−1:
            for lane in 0 .. 31:                       # all lanes, same step
                g = lane·steps + k
                need[lane] = false
                if g < here:
                    slot = x[lane] & (2^n − 1)
                    s = symbol of slot; byte(g, b) = s
                    x[lane] = f[s] · (x[lane] >> n) + slot − cum[s]
                    need[lane] = x[lane] < 2^16
            for lane in 0 .. 31 in increasing order:   # renormalise
                if need[lane]:
                    x[lane] = (x[lane] << 16) | stream[ptr]; ptr += 1
    valid iff ptr == N and x[lane] == 2^16 for all 32 lanes

A word index `≥ N` (reading past the stream), a final state other than 2^16
in any lane, or `ptr ≠ N` makes the run corrupt; the engine counts such lanes
and fails the read.

Encoding (informative): each lane starts at `x = 2^16` and the planes and
steps are walked in reverse; for a live symbol s, if `x ≥ f[s] << (32 − n)`
the lane emits `x & 0xFFFF` and sets `x >>= 16`, then
`x = ((x / f[s]) << n) + (x % f[s]) + cum[s]`. The words emitted at one step
are placed in increasing lane order, and the blocks of successive (reverse)
steps are placed from the end of the stream towards its start, so the decoder
reads it forwards.

The rANS streams of different layers are independent (their own states and
words); a reader that does not want a layer skips its bytes.

### 4.9 Words and transforms

Source: `athz_common.slang` (`athzForward`, `athzStep`, `athzCombine`,
`athzWithin`, `athzUnwithin`, `athzPreviousWord`), `athz_decode.slangh`.

The four bytes of word w of an element (planes 4w .. 4w+3) form the residual
`r = b0 | b1 << 8 | b2 << 16 | b3 << 24`. The word's transform t (plane entry
of plane 4w+3, bits 0–7) says how the value v is recovered. Transforms are per
word and per kind, for the whole asset.

| t | Name | Residual the writer coded | Decoding |
|---:|---|---|---|
| 0 | none | `v` | `v = r` |
| 1 | delta | `zigzag(v − v_prev)` | `v_g = Σ_{j ≤ g} unzigzag(r_j)` mod 2^32 along the run |
| 2 | delta16 | `zigzag16x2(sub16x2(v, v_prev))` | each half: `Σ_{j ≤ g} unzigzag16(r_j half)` mod 2^16 |
| 3 | xor | `v XOR v_prev` | `v_g = r_0 XOR … XOR r_g` |
| 4 | parent | `zigzag(v − p)` | `v = p + unzigzag(r)` mod 2^32 |
| 5 | parent16 | `zigzag16x2(sub16x2(v, p))` | `v = add16x2(p, unzigzag16x2(r))` |
| 6 | xorParent | `v XOR p` | `v = p XOR r` |
| 7 | within | `within(v, q)` | `v = unwithin(r, q)` |
| 8 | parentWithin | `within(sub16x2(v, p), sub16x2(q, p_q))` | `v = add16x2(unwithin(r, sub16x2(q, p_q)), p)` |

- `v_prev` is the same word of the previous element of the same run; the
  run's first element (g = 0) is predicted from 0. Runs are independent.
- `p` is word w of the element's parent (§4.9), 0 when the element has
  no parent (band 0). For t = 8 without a parent, the word is decoded as
  t = 7.
- `q` is the element's own previous word in coding order,
  `pw = tailWord` for w = 0, `pw = tailWord − 1` for w = tailWord + 1,
  otherwise `pw = w − 1`; `p_q` is the parent's word pw.
- `within(v, q) = zigzag16((v.lo − q.hi) mod 2^16) | zigzag16((v.hi − v.lo)
  mod 2^16) << 16`, where `.lo`/`.hi` are the low/high halves.
  `unwithin(r, q)`: `lo = (unzigzag16(r.lo) + q.hi) mod 2^16`,
  `hi = (unzigzag16(r.hi) + lo) mod 2^16`, `v = lo | hi << 16`.
- The tail word takes only transforms 0 to 3 (a file with a higher one is
  refused): it is decoded first and identifies the parent of the remaining
  words.

The writer only gives a word a transform under which each byte depends on
bytes of layer G or of its own layer (informative; transforms 0, 3, 6 are
always allowed; 1, 4 need every byte of the word to be allowed to depend on
the bytes below it; 2, 5 on each half's low byte; 7, 8 additionally on the
previous word's high half). A reader that leaves out layers can therefore still
decode every byte of the layers it keeps.

#### Parents

Source: `athz_decode.slangh`, `athz_encode.slang` (`athzParents`).

After an element's tail word is decoded:

- Splat (band `levels`): the parent is group `min(tail, finestGroups − 1)` of
  the finest merged level.
- Merged group of band l > 0: with `cell = tail >> 3`, the parent is group
  `min(a, groups(l−1) − 1)` of level l−1, where `a` is the first index whose
  cell code (tail word) in level l−1 is `≥ cell` (lower bound over the
  ascending cell codes).
- Band 0: no parent.

The parent's words are its decoded coded-element words: for field 8 under
CPCA, the parent's CPCA code words, not its decoded transfer; for fields 11
and 12, 0. Merged levels must therefore be decoded coarsest first, and the
splats after the finest level.

### 4.10 Decoding a page (summary)

1. Check the page entry (count, band, layer sizes) and its CRC over the
   present layers' bytes.
2. For each layer L to decode, for each run r: locate the run (`runEnd`), read
   `N`, the 32 states and the stream where `hasRans(k, L)`, and the raw part.
3. For i = 0..P−1 in coding order: obtain byte `b = planeAt(i)` of every
   element of the run from its layer's rANS stream or raw block. After each
   word's fourth plane, undo the word's transform for all elements of the run;
   after the tail word, determine each element's parent.
4. The page's element g of run r is element `firstOfPage + r·R + g` of its
   band.
5. Under CPCA, turn field 8 (the code) into the transfer (§4.12).

A layer not decoded leaves its bytes zero in the engine; G is always decoded.

### 4.11 Finest level's starts

At `startsOffset`, `finestGroups` u32: the first splat (store index) of each
group of the finest merged level. Stored raw.

### 4.12 Layer T as CPCA

Source: `athz_cpca.slang` (`athzCpca`, `athzCpcaPack`, `athzCpcaUnpack`),
`Athz.cpp` (`bindCpca`, `writeAthz`, `decodeAsset`).

Where `codedTransferWords = 4` (version 3+), field 8 of every element (both
kinds) is a 4-word code in place of the transfer, coded through the same
planes and transforms (its parent prediction uses the parent's code).
Constraints checked by the reader: `codedTransferWords = 4`,
`transferCount = 112`, `1 ≤ cpcaClusters ≤ 256`, `cpcaComponents = 8`, the
basis within the file. The writer accepts only a basis of exactly C = 256
clusters and K = 8 (1 171 468 bytes); the reader does not check `basisBytes`
against C and K.

Code layout (4 words, 128 bits, bit j is bit `j % 32` of word `j / 32`):

| Bits | Content |
|---|---|
| 0–7 | cluster id |
| 8 + 4i .. 11 + 4i, i = 0..23 | weight i = b·K + a (block b = 0..2, axis a = 0..7) as `q + levels`, 0 to 2·levels |
| 104–127 | 0 |

So word 0 holds the id and weights 0–5, word 1 weights 6–13, word 2 weights
14–21, word 3 bits 0–7 weights 22–23.

Basis (f32 array at `basisOffset`; C clusters, K axes, blocks of
`D = (16, 48, 48)` coefficients, 112 in all):

| Float index | Count | Content |
|---|---|---|
| 0 | 112·C | centroid c at `112c`, in normalised coefficients (writer only) |
| 112·C | 3 | per-block norm (writer only) |
| `meanAt[b]` | C·D_b | block b's means: cluster c, coefficient d at `meanAt[b] + c·D_b + d` |
| `meanAt[b] + C·D_b` | C·D_b·K | axes: `+ c·D_b·K + d·K + a` |
| `meanAt[b] + C·D_b + C·D_b·K` | C·K | scales: `+ c·K + a` |

`meanAt[0] = 112C + 3`; `meanAt[b+1] = meanAt[b] + C·D_b + C·D_b·K + C·K`.

Decoding an element's code into the transfer (`transferWords` words, f32
arithmetic):

    id = min(code & 0xFF, C − 1)
    for b in 0..2, with j running over 0..111 (block 0: j = d; 1: 16 + d; 2: 64 + d):
        for a in 0..7:
            nib = (code >> (8 + 4(bK + a))) & 0xF
            w[a] = (nib − levels) · scale[b][id][a] / levels
        for d in 0..D_b−1:
            v = mean[b][id][d]
            for a in 0..7: v += axes[b][id][d][a] · w[a]
            h[j] = f16(clamp(v, −65000, 65000))
    transfer word m = h[2m] | h[2m+1] << 16,   m = 0..55

The encoder (informative): the id is the centroid nearest in squared distance
to `coefficient_j / norm[b]`; weight `w_a = Σ_d axes[d][a] (coefficient_d −
mean_d)`, `q = clamp(round(w_a / scale_a · levels), −levels, levels)`.

### 4.13 Layer K

Source: `Athz.cpp` (`writeAthz`, `readAthzSkinned`), `athz_interleave.slang`
(`athzToHalves`, `athzFromHalves`), `SplatSkinner.h`.

Version 4. Present when `influenceWords = 8` (header flag 4, layerMask bit
8). Per element, fields 11 and 12 (§4.6) are coded in the pages like
any other field, in layer K; the splats are in store order, so element i's rig
is splat i's. Merged groups carry zeros there.

The frames block at `temporalOffset`, `temporalBytes = 64 + 8F + 24FJ`:

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 64 | f32[16] | geomBind | the geometry's bind transform, 16 floats as `SplatSkinning::geomBindTransform` holds them (row-major as USD's GfMatrix4d) |
| 64 | 8F | f64[F] | times | each frame's time code |
| 64 + 8F | 24FJ | u32[F][J][6] | joints | frame-major, then joint: 12 f16, halves `h[0..11]`, word k = `h[2k] | h[2k+1] << 16` |

A joint's 12 halves are rows 0 to 2 of the 4×4 matrix the skinner takes
(USD's joint matrix transposed: translation in the last column), element
`(row, col)` at half `4·row + col`; row 3 is `0 0 0 1` and is not stored.
Decoding: each half to f32, row 3 restored. The time codes are in the stage's
units; the file does not store time codes per second.

### 4.14 Prefixes

Source: `Athz.cpp` (`prefixPages`, `writeAthzPrefix`, `writeAthzPrefixBytes`,
`readAthz`, `readAthzCut`, `AthzSource`).

A prefix holds the first `keep` pages of a file (`keep` ≤ the total, any
number): the header's `pages = keep`, every section cut to its first `keep`
pages' bytes (`sectionBytes[L]` = original `offset[L]` of page `keep`), and
the page table cut to `keep` entries. Everything else is as in the source: the
level entries, `count`, `gaussians`, plane entries, tables, starts, basis,
frames, extensions (bytes 64–4095 copied as they are) and `assetId`. The
regions are laid out again (§4.2) and the asset table is rewritten in
the version 4 layout with header version 4, whatever the source's version.

A reader distinguishes:

- `pages = leafFirstPage + leafPages`: the whole asset.
- `leafFirstPage ≤ pages < total`: every merged level is whole; splat pages
  `pages − leafFirstPage` onwards are absent. The engine reads it as a cloud
  whose missing chunks are not resident (drawn as their merged groups).
- `pages < leafFirstPage`: a fixed cut. Only the finest band whose pages are
  all present is usable, decoded with the levels above it as parents; the
  engine refuses it as a cloud to stream, and refuses it entirely if not even
  level 0 is whole.

A skinned file is read with layer K only when whole.

### 4.15 Versions

Source: `Athz.cpp` (`kVersion`, `decodeAsset`).

| Version | Asset table | Plane entry word 3 | Adds |
|---|---|---|---|
| 1 | offsets 0–71, then from `boundsLo` on (at 72) | encoder estimate, read as transform 0 | — |
| 2 | as version 1 | transform | word transforms |
| 3 | offsets 0–107, then from `boundsLo` on (at 108) | transform | curvature, CPCA |
| 4 | as tabulated above (`boundsLo` at 140) | transform | layer K |

Fields absent from a version read as 0. In version 1 and 2 tables every field
from `boundsLo` onward sits 68 bytes earlier than in version 4; in version 3,
32 bytes earlier.

### 4.16 What a reader must refuse

Source: `Athz.cpp` (`decodeHeader`, `decodeAsset`, `parse`, `checkPage`),
`athz_decode.slangh`.

- A file shorter than 4096 bytes; no `ATHZ` magic; version outside 1 to 4;
  `headerBytes ≠ 64`; a header CRC mismatch.
- A flag bit outside 0x3F; a `layerMask` without bit 1 or with bits outside
  0xF; R = 0, not a multiple of 32 or > 256; `pageRuns = 0`; `entryBytes ≠ 64`.
- An asset table past the end of the file or shorter than its counts;
  `bytes ≠ assetBytes`; `kinds ≠ 2`; `levels` 0 or > 30; `count = 0`;
  `shWords` 0 or > 64; `planes ≠ 4W`; `tableLog` outside 9 to 12;
  `chunkSplats ≠ R × pageRuns`; `gaussians ≠ count`.
- CPCA: `codedTransferWords ≠ 0` with `codedTransferWords ≠ 4`,
  `transferCount ≠ 112`, `cpcaClusters` 0 or > 256, `cpcaComponents ≠ 8`, or a
  basis outside the file.
- A rANS table whose frequencies do not sum to `2^tableLog`.
- `curvatureWords` not 0 or 3.
- Layer K: `influenceWords` not 0 or 8; `gradientWords` not 0 or 3;
  `influenceWords ≠ 0` and `frames ≠ 0` disagreeing; a frames block outside
  the file; `temporalBytes ≠ 64 + 8F + 24FJ` where F ≠ 0.
- A plane entry with transform > 8, or > 3 on the tail word's planes; encoding
  > 1; width > 8; layer > 3; a rANS table index ≥ `tables`; a layer not in
  `layerMask`.
- A level entry that breaks §4.5's checks; `startsBytes ≠
  4 × finestGroups`; starts outside the file; a section outside the file.
- `pages` greater than the asset's total; a page table outside the file.
- A page whose band or count is not the one its index implies; nonzero bytes
  in an absent layer; for a present layer, `offset + bytes > sectionBytes` or
  `bytes < 4 × pageRuns`.
- An extension record running past `assetOffset`.
- A page whose CRC does not match, before any of it is decoded.
- A run that decodes as corrupt (§4.8, rANS).

The engine does not otherwise bound-check `runEnd` values or `rawStart`
against a page's bytes; an independent reader should.

### 4.17 Not determined from the code

- f32 → f16 conversions (page boxes before their outward step, CPCA transfer
  values, layer K joint halves) use the GPU backend's `f32tof16`; the rounding
  mode is not set by the code (round-to-nearest-even is the usual hardware
  behaviour). The CPCA reconstruction sums in f32 in the order given; whether
  the compiler fuses multiply-adds is not controlled, so bit-exact agreement of
  a CPU reimplementation with the engine's decoded transfer is not guaranteed.
- Version 1 and 2 files written by earlier encoders: the current reader
  decodes every version with the lane assignment (lane `g / steps`) and plane
  coding order (tail word first) above. `docs/decisions.md` describes the
  first encoder's lane assignment as `g % 32`; whether files of versions 1 to
  2 written before the change exist with that layout, and how they would be
  read, cannot be determined from the current code.
- The bit layout of a cell code (the Morton interleave of the cell's
  coordinates) is defined by the LOD builder, not by this format; a reader
  needs only that codes ascend within a level and that a parent's code is
  `cell >> 3`.
- The meaning of the source extension's bytes as text (encoding) is not
  specified.

## 5. `.athv`: a video of gaussians

Written by `athenea video-encode`; read by `AthvReader`
(`modules/video/include/athenea/video/Athv.h`, `modules/video/src/Athv.cpp`).
A frame's splats are put in the luma (and optionally chroma) planes of a
picture a hardware HEVC encoder takes; what the codec cannot carry
exactly travels beside it in the same record. The file is a header and then
one record a picture, each whole in itself, so a sender can cut it into
packets and a receiver put records back together without the file.

### 5.1 Header: 128 bytes (version 2)

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 4 | u32 | magic | `ATHV` (0x56485441) |
| 4 | 4 | u32 | version | 2 (1 is read, §5.4) |
| 8 | 4 | u32 | codec | FourCC, `hvc1` (0x31637668) |
| 12 | 4 | u32 | bits | bits a sample: 8 or 10 |
| 16 | 4 | u32 | restPerColour | 0, 3, 8, 15 |
| 20 | 4 | u32 | chunkSplats | splats a chunk: a multiple of 32768 |
| 24 | 4 | u32 | columns | tiles a row of the mosaic |
| 28 | 4 | u32 | maxSplats | the most splats a picture holds: what its tiles are sized for |
| 32 | 4 | u32 | flags | below |
| 36 | 4 | u32 | rendition | this file's id among renditions of the same frames |
| 40 | 8 | f64 | rate | frames a second |
| 48 | 4 | u32 | rateNumerator | the rate as a fraction |
| 52 | 4 | u32 | rateDenominator | |
| 56 | 4 | u32 | sourceFrames | source frames in the file: the loop's length |
| 60 | 4 | f32 | quality | the encoder's constant quality; 0: by bit rate |
| 64 | 4 | u32 | positionBlock | 0: positions over the frame's range; else splats a box (a power of two, 64 to chunkSplats) |
| 68 | 4 | u32 | positionHighBits | the position's bits past `bits`, 0 to `bits` |
| 72 | 4 | f32 | positionTarget | > 0: adaptive precision, metres |
| 76 | 4 | u32 | geometryInterval | geometry layer: frames from key to key; 0 every frame a key |
| 80 | 4 | u32 | geometryBits | bits 0–7 position bits (10–16), 8–15 log-scale bits (6–10), 16–23 the prediction order (1, 2, 3), 24–31 rotation bits (5–10; 0 read as 10); 0: the defaults 16, 10, 1, 10 |
| 84 | 4 | u32 | geometryFold | with flag bit 7: the folds, bits 0–7 position, 8–15 log scale, 16–23 rotation |
| 88 | 4 | f32 | volumeDensityScale | with flag bit 8 |
| 92 | 36 | — | reserved | zero |

**Flags.**

| Bit | Value | Meaning |
|---|---|---|
| 0 | 1 | the position high parts are planes of the picture, not side data |
| 1 | 2 | the source was fitted to be drawn antialiased (Mip-Splatting's compensation) |
| 2 | 4 | colours in chroma: YCoCg, Co and Cg in the picture's second plane |
| 3 | 8 | stable order: index i is the same gaussian in every frame |
| 4 | 16 | the geometry layer: position, scale and rotation exact beside the video |
| 5 | 32 | the geometry layer refreshes blocks of 256 in turn between keys |
| 7 | 128 | geometry in the video: position, scale and rotation as luma planes |
| 8 | 256 | a medium: the splats are gaussians of extinction (§5.9) |

The writer writes the header, the records, and the header again at the end
with `maxSplats` and `sourceFrames`.

### 5.2 Record

Records follow the header back to back; each length is a multiple of 4.
The first 28 bytes are the same in every message a stream carries (§5.11),
so a relay routes on them without reading further.

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 4 | u32 | magic | `ATVF` (0x46565441) |
| 4 | 2 | u16 | type | 0: a picture |
| 6 | 2 | u16 | rendition | the header's |
| 8 | 4 | u32 | frame | the presentation frame: in a file, the source frame |
| 12 | 8 | u64 | ptpTime | ns; 0 in a file, the send time in a stream |
| 20 | 4 | u32 | bytes | the whole record |
| 24 | 2 | u16 | version | 2 |
| 26 | 2 | u16 | flags | bit 0 a key picture (an HEVC IRAP); bit 1 a geometry key (every gaussian absolute) |
| 28 | 4 | u32 | index | the picture's index in the file |
| 32 | 4 | u32 | source | the source frame it is a slice of |
| 36 | 4 | u32 | slice | which of that frame's pictures |
| 40 | 4 | u32 | slices | the frame's pictures |
| 44 | 4 | u32 | firstSplat | where in the frame its splats start (Morton rank) |
| 48 | 4 | u32 | splats | the picture's splats |
| 52 | 4 | u32 | sourceSplats | the frame's splats |
| 56 | 8 | f64 | time | the source's time code, seconds |
| 64 | 16 | char[16] | timecode | `HH:MM:SS:FF`, NUL-terminated; a label |
| 80 | 4 | u32 | rangeCount | floats in the ranges (two an attribute; at most 4096) |
| 84 | 4 | u32 | highBytes | the high parts' bytes, inflated (0: none) |
| 88 | 4 | u32 | highStored | their bytes as stored |
| 92 | 4 | u32 | parameterSets | 0, or 3 on a key picture (VPS, SPS, PPS; at most 16) |
| 96 | 4 | u32 | unitBytes | the access unit's bytes |
| 100 | 4 | u32 | geometryStored | the geometry layer's bytes as stored (0: none) |
| 104 | | f32 × rangeCount | ranges | |
| | | bytes | high | `highStored` bytes, a zlib stream (RFC 1950); padded to 4 |
| | | bytes | geometry | `geometryStored` bytes, a zlib stream; padded to 4 |
| | | | parameter sets | each a u32 length and its bytes, padded to 4 |
| | | bytes | unit | the access unit: NAL units, each after a 4-byte big-endian length; padded to 4 |

The high parts inflate to exactly `highBytes`; the geometry layer to
`⌈splats / chunkSplats⌉ · chunkBytes`, where `chunkBytes` is
`16 · chunkSplats`, plus `chunkSplats / 256 · 32` with prediction order 3.
A record whose lengths run past its end, or whose data do not inflate to
those sizes, is refused. A picture never names a source frame past the one
after the last seen: a reader groups slices by `source` as it walks the
file.

A decoder takes the access unit as HEVC with `hvcC`-style length prefixes;
an Annex B stream is the parameter sets and the NAL units, each after a
start code `00 00 00 01` (`annexB` in `Athv.h`).

### 5.3 Pictures

A picture is one frame's splats, or one slice of them, laid out as a single
luma image (the *mosaic*). With `chroma`, it also has an NV12 second plane.
Each attribute of every splat is one sample in one tile of the mosaic. This
section covers what the samples mean. The header and record fields named
below (`bits`, `restPerColour`, `chunkSplats`, `columns`, `maxSplats`, the
flags, `positionBlock`, `positionHighBits`, `positionTarget`, the geometry
fields, `geometryFold`, `volumeDensityScale`, and the record's `splats`,
`firstSplat`, `source` and `ranges`) are the container's.

All multi-byte values are little endian. Unless stated otherwise, arithmetic
is unsigned 32-bit and `>>` on a signed value is arithmetic.

#### Splat order

Source: `shaders/athenea/scene/video_planes_encode.slang` (`videoPlanesMorton`), `shaders/athenea/common/morton.slang`.

Within a frame, splats are numbered by their *Morton rank* `r`. The encoder
computes a 30-bit key for each splat over the position ranges (attributes 0–2
of the ranges in use):

```
u      = saturate((p - lo) / max(hi - lo, 1e-20))       per axis
q      = min(uint(u * 1024), 1023)                       per axis
key    = spread(q.x) << 2 | spread(q.y) << 1 | spread(q.z)
```

It then radix-sorts the splats on the key's 30 bits. A picture holds Morton
ranks `firstSplat .. firstSplat + splats - 1` of its frame. Inside the
picture, splat `i = r - firstSplat` is the picture-local rank, and the decoder
writes it to the frame's gaussian `firstSplat + i`. The decoder does not need
the key.

The `stableOrder` flag (`--order stable`) means one sort, taken on the first
frame, is reused for every frame. Index `r` is then the same gaussian
throughout the file. The geometry layer's predicted frames depend on this.

#### Layout constants

Source: `modules/video/include/athenea/video/Planes.h` (`PlaneLayout`), `modules/video/src/Planes.cpp` (`PlaneLayout::make`).

| Constant | Value |
|---|---|
| tile width | 512 samples |
| block | 64 × 64 samples = 4096 splats |
| blocks a tile row | 8 |
| band (smallest chunk) | 8 blocks = 32768 splats = 64 rows of a tile |
| `chunkSplats` | a multiple of 32768; `video-encode` always writes 32768 |
| largest picture | width ≤ 16384, height ≤ 16384, width × height ≤ 8192² |

#### Mosaic size

Source: `modules/video/src/Planes.cpp` (`PlaneLayout::make`).

`planes` is the number of luma planes, given under "Planes by mode". A
decoder builds every picture's layout from the record's `splats` and the
header's `maxSplats`, `chunkSplats` and `columns`:

```
chunks      = ceil(splats / chunkSplats)
tileChunks  = max(chunks, ceil(maxSplats / chunkSplats))
tileHeight  = tileChunks * (chunkSplats / 512)          rows; a multiple of 64
columns     = header.columns
width       = columns * 512
height      = ceil(planes / columns) * tileHeight
```

Because the tiles are sized for `maxSplats`, every picture in a stream has the
same dimensions. No further rounding is applied. The height is always a
multiple of 64, so it is even, as NV12 requires.

The encoder chooses `columns`. For each `c` in `1 .. planes`, it computes
`w = 512 c` and `h = ceil(planes / c) × tileHeight`. It discards any `c` with
`w` or `h` above the side limit (16384, or the smaller `--max-picture`) or
`w × h` above 8192². Of the rest, it keeps the `c` with the smallest
`max(w, h)`, and on a tie the smallest `w × h`. `--columns` forces a value.

#### Splat to sample

Source: `shaders/athenea/scene/video_planes_common.slang` (`pixelOf`, `planeAndSplatOf`).

Plane `p` is the tile at column `p mod columns` and row `p / columns`, so the
tiles fill the mosaic row-major. The sample of splat `i` in plane `p` is at:

```
block = i >> 12                  j = i & 4095
x = (p mod columns) * 512 + (block mod 8) * 64 + compact2(j)
y = (p / columns) * tileHeight + (block / 8) * 64 + compact2(j >> 1)
compact2(v): bits 0, 2, 4, 6, 8, 10 of v packed into bits 0..5
```

Inside a 64 × 64 block, splats follow Z order, with x taking the even bits of
`j` and y the odd bits. Blocks go eight to a tile row, so chunk `c` of a
picture (splats `c·chunkSplats ..`) is rows `c·chunkSplats/512 ..` of every
tile. Splats `4m .. 4m+3` occupy the 2 × 2 quad whose top-left sample is at
even (x, y).

Fill values:

- A sample of a real plane whose index `i ≥ splats` repeats splat
  `splats − 1`.
- A tile whose index is `planes` or more (the spare tiles of the last row) is
  mid grey: `1 << (bits − 1)`.

#### Sample storage

Source: `modules/video/src/Planes.cpp` (`setLayout`), `video_planes_encode.slang` (`videoPlanesEncode`), `video_planes_decode.slang` (`sampleAt`).

| `bits` | Luma storage | Sample value `s` |
|---|---|---|
| 8 | one byte a sample (HEVC Main) | `s` = the byte |
| 10 | one 16-bit word a sample, P010: the value in the top 10 bits | `s` = word >> 6 |

With `chroma` (8 bits only), the second plane is NV12's: `width/2 × height/2`
pairs of bytes `(Cb, Cr)`, rows `width` bytes apart. Pair `(cx, cy)` covers
luma samples `(2cx .. 2cx+1, 2cy .. 2cy+1)`, which are four consecutive splats
`4m .. 4m+3` of one tile. Pairs under tiles that are not a colour Y plane,
and pairs whose first splat `4m ≥ splats`, are `(128, 128)`.

### 5.4 Attributes

Source: `shaders/athenea/scene/video_planes_common.slang` (`plainAttributeOf`, `attributeOf`), `shaders/athenea/common/packing.slang`.

A splat has `attributes = 14 + 3 × restPerColour` scalar attributes
(`kAttributesBeforeRest = 14`), where `restPerColour` is one of 0, 3, 8 or 15.
The values are those of the engine's packed store, read as follows.

| Index | Attribute | Value as sent |
|---|---|---|
| 0, 1, 2 | position x, y, z | float32, in the cloud's space |
| 3 | opacity | linear, as stored (a 3DGS logit has already been through the sigmoid); normally 0..1 |
| 4, 5, 6 | scale x, y, z | the natural log of the standard deviation (the store's f16 `ln scale`), clamped to [−20, 20] |
| 7, 8, 9, 10 | rotation x, y, z, w | the store's quaternion decoded (`decodeQuaternion`) and negated when `w < 0`, so `w ≥ 0`; not renormalised |
| 11, 12, 13 | base colour r, g, b | the store's f16 base colour (see below) |
| 14 + k | rest coefficient k | `k = basis × 3 + colour` (colour 0 r, 1 g, 2 b; basis 0 .. restPerColour−1 in 3DGS's order, degree 1 first) |

What the stored values mean:

- **Base colour.** For an SH source this is `0.5 + C0 · f_dc`, with
  `C0 = 0.28209479177387814`. For other sources it is the source's colour as
  the loader decoded it (`splat_encoding.slang`, `decodeColour`). The format
  does not state a transfer function. The base colour is not clamped and can
  leave 0..1.
- **Rest coefficients.** They are in the store's rgb-per-basis order, not a
  3DGS PLY's colour-major order. Coefficients past the source's own count are
  0.
- **Rotation word.** The store's rotation is smallest-three: the dropped
  component's index (0 x, 1 y, 2 z, 3 w) in bits 0–1. The other three, in
  x, y, z, w order without the dropped one, are at bits 22, 12 and 2, ten
  bits each, as `round((c / √½ · 0.5 + 0.5) · 1023)`.

With `chroma`, colour attributes are sent as YCoCg instead of RGB. Colour
quantity `q` is the base colour when `q = 0` and harmonic basis `q − 1`
otherwise. Its three attributes `11 + 3q`, `12 + 3q` and `13 + 3q` hold Y, Co
and Cg of that quantity's (r, g, b):

```
Co = R − B,   t = B + Co/2,   Cg = G − t,   Y = t + Cg/2          (float32, no rounding)
inverse:  t = Y − Cg/2,  G = Cg + t,  B = t − Co/2,  R = B + Co
```

#### Planes by mode

Source: `video_planes_common.slang` (`lowPlane`, `attributePlane`, `planeAttribute`), `Planes.h` (`PlaneLayout::planes`).

Let `C = 3 + 3 × restPerColour` without `chroma`, or `C = 1 + restPerColour`
with it, be the number of colour planes.

**Plain** (no geometry layer, no geometry in the video):

| Planes | Content |
|---|---|
| 0–2 | position high parts x, y, z (only with `hiInVideo`) |
| 0–2, or 3–5 with `hiInVideo` | position low parts x, y, z |
| next 8 | attributes 3–10: opacity, log scale x y z, rotation x y z w |
| without `chroma`: next `3 + 3·restPerColour` | attributes 11, 12, … in index order |
| with `chroma`: next `1 + restPerColour` | quantity q's Y (attribute `11 + 3q`); its Co and Cg are in the chroma under that tile |

`planes = (hiInVideo ? 6 : 3) + 8 + C`. Without `chroma` this equals
`(hiInVideo ? 6 : 3) + attributes − 3`.

**Geometry layer** (flags bit 4): the planes are opacity (plane 0), then the
colour planes as above. `planes = 1 + C`. Position, scale and rotation travel
in the geometry layer.

**Geometry in the video** (flags bit 7): see "Geometry in the video".
`planes = 9 + 2F + 1 + C`, where `F` is the number of folded geometry
attributes.

### 5.5 Quantisation

Source: `video_planes_common.slang` (`quantise`, `dequantise`, `splitPosition`, `joinPosition`), `video_planes_decode.slang`.

Every attribute is quantised over a range `[lo, hi]` with `L` levels:

```
quantise(v, lo, hi, L)   = hi > lo ? min(round(saturate((v − lo) / (hi − lo)) · L), L) : 0
dequantise(q, lo, hi, L) = hi > lo ? lo + (hi − lo) · (min(q, L) / L) : lo         (float32)
```

`round` is the shader's: to nearest, with ties resolved however the GPU
target's `round` resolves them. The code does not pin this down.

#### Ranges

Source: `video_planes_encode.slang` (`videoPlanesRangePartial`, `videoPlanesRangeFinish`, `videoPlanesRangeWiden`), `apps/athenea/src/CmdVideo.cpp`.

A record's `ranges` holds two float32 for each attribute: `ranges[2a] = lo`
(the minimum) and `ranges[2a+1] = hi` (the maximum) of attribute `a`. There
are `2 × attributes` floats.

- **Basis.** The values are those in the attributes table, after the log-scale
  clamp, the rotation's `w ≥ 0`, and YCoCg with `chroma`. With geometry in the
  video, attributes 7–10 are the canonical rotation's components.
- **No splats.** A frame with no splats gives `(0, 0)`.
- **Range groups.** By default (`--ranges group`) the ranges are the union
  over a range group: `--range-group` frames, or the key interval when that is
  not set. Every record in the group repeats the same ranges. `--ranges frame`
  uses each frame's own.
- **Geometry layer.** A geometry layer sent as differences (`geometryInterval`
  ≠ 1) requires one range group for the whole sequence.

Every attribute that is a plane is decoded as
`dequantise(s, lo_a, hi_a, L)`, with `L = 2^bits − 1` (255 or 1023).
Positions are the one exception.

#### Positions in planes

Source: `video_planes_encode.slang` (`positionOf`, `sampleOf`, `videoPlanesHigh`), `video_planes_decode.slang` (`videoPlanesDecode`, `highAt`, `positionRangeAt`, `highBitsAt`).

This section applies in plain mode. Each axis of a position is quantised to
`bits + H` bits:

```
Lp   = 2^(bits + H) − 1
Q    = quantise(p_axis, lo, hi, Lp)
high = Q >> bits
low  = Q & (2^bits − 1)
lowSample = (high odd) ? (2^bits − 1) − low : low        the low part is folded (mirrored) on odd high parts
```

The decoder rebuilds the axis:

```
Q = (high << bits) | ((high odd) ? (2^bits − 1) − min(low, 2^bits − 1) : min(low, 2^bits − 1))
p_axis = dequantise(Q, lo, hi, Lp)
```

Where each input comes from:

| Mode | `H` | `[lo, hi]` | high part |
|---|---|---|---|
| `hiInVideo` | `bits` | the record's range of the axis | planes 0–2 (samples) |
| side, `positionBlock = 0` | `positionHighBits` (0 .. `bits`) | the record's range of the axis | side payload |
| side, `positionBlock > 0` | `positionHighBits` | the block's box (below) | side payload |
| side, `positionBlock > 0`, `positionTarget > 0` | the block's own byte (adaptive) | the block's box | side payload |

With `positionHighBits = 0`, no high parts are stored, `high = 0`, and the
low plane holds `Q` unfolded. `hiInVideo` requires `positionBlock = 0` and
`positionHighBits = bits`.

**Position blocks.** When `positionBlock > 0`, its value is a power of two
from 64 to 32768. Block `b` of a chunk covers that chunk's splats
`b·positionBlock .. (b+1)·positionBlock − 1`, a run of the Morton order. Its
box, per axis, is the smallest f16 `lo` and f16 `hi` that enclose the run's
positions:

- `lo = halfBelow(min)`: the f16 nearest the minimum, stepped down one ulp if
  it lies above.
- `hi = halfAbove(max)`: likewise, stepped up if it lies below.

Positions in the block are quantised over the box, read back as float32. A
block with no splats has the box `(0, 0)`. With `--hold-boxes`, the box is the
union over the range group's frames instead of the frame's own box. It is
stored the same way.

**Adaptive precision.** When `positionTarget > 0`, block `b` carries its own
high-bit count `H_b`, computed as follows:

```
widest = max over the three axes of (f16 hi − f16 lo)
H_b    = widest > positionTarget ? min(ceil(log2(widest / positionTarget)), positionHighBits) : 0
```

The storage of the high parts does not change with `H_b`. Each entry keeps
its size, and the unused bits are zero.

#### Chroma samples

Source: `video_planes_encode.slang` (`videoPlanesChroma`, `chromaWeightOf`), `video_planes_decode.slang` (`colourAt`).

For colour quantity `q`, the Y plane is per splat. The chroma pair at a quad
holds the weighted mean of the four splats' Co and Cg. Splats `≥ splats` are
left out of the mean.

```
w_s = 1                                                   (chromaWeight 0, the default)
w_s = max(opacity, 1e-4) · exp(ls_x + ls_y + ls_z − min(ls))   (encoder option --chroma-weight area)
Cb  = quantise(Σ w_s Co_s / Σ w_s, lo_{12+3q}, hi_{12+3q}, 255)
Cr  = quantise(Σ w_s Cg_s / Σ w_s, lo_{13+3q}, hi_{13+3q}, 255)
```

The chroma weight is an encoder choice and is not recorded in the file. The
decoder takes, for each splat, Y from its own sample, and Co and Cg from the
pair at `(x/2, y/2)` of that sample. It dequantises the three over their
ranges and converts them to RGB with the inverse YCoCg.

#### Decoded splat

Source: `video_planes_decode.slang` (`videoPlanesDecode`, `writeRest`).

The decoder writes these values in every mode:

| Field | Value |
|---|---|
| opacity | `clamp(dequantised attribute 3, 0, 5)` |
| log scale | as decoded, stored as f16, clamped to ±65000 |
| rotation (plain mode) | `q = (a7, a8, a9, a10) / |q|` (`(0,0,0,1)` when `|q| ≤ 1e-8`), re-encoded smallest-three |
| colours | as decoded, f16 |
| rest coefficients past `restPerColour` | 0 |

### 5.6 Side data

Each record carries:

- `ranges`;
- the deflated *high* payload (container fields `highRaw` / `highStored`);
- the deflated *geometry* payload (`geometryStored`).

Both payloads are zlib streams, written with `compress2` at level 9.

#### High payload

Source: `Planes.h` (`blockBytes`, `bitsBytes`, `highChunkBytes`), `video_planes_encode.slang` (`videoPlanesBlocks`, `videoPlanesBlockBits`, `videoPlanesHigh`), `video_planes_decode.slang` (`highAt`).

The inflated payload is `ceil(splats / chunkSplats)` chunks of
`highChunkBytes` each. The chunks are the picture's own, and chunk `c` covers
picture splats `c·chunkSplats ..`. Let `nb = chunkSplats / positionBlock` and
`sb = 1` at 8 bits, `2` at 10 bits. The payload is empty with `hiInVideo` and
no `positionBlock`, with the geometry layer, and with geometry in the video.

| Offset in the chunk | Bytes | Present when | Content |
|---|---|---|---|
| 0 | `12 · nb` | `positionBlock > 0` | the boxes: for block `b` and axis `a`, the u32 at word `3b + a` is `f16 lo \| f16 hi << 16` |
| `12 · nb` | `ceil(nb / 4) · 4` | `positionBlock > 0` and `positionTarget > 0` | `H_b`: one byte per block, block `b` at byte `b`; zero padding |
| after the above | `3 · chunkSplats · sb` | not `hiInVideo` and `positionHighBits > 0` | the high parts, axis-major: entry `e = axis · chunkSplats + (i mod chunkSplats)` is a u8 (8 bits) or u16 (10 bits) at byte `e · sb` |

```
highChunkBytes = (positionBlock ? 12·nb : 0)
               + (positionTarget > 0 && positionBlock ? ceil(nb/4)·4 : 0)
               + (!hiInVideo && positionHighBits ? 3·chunkSplats·sb : 0)
```

Entries for splats `≥ splats` are 0.

At 10 bits a high part takes two bytes. Comments in `Planes.h` and
`video_planes_common.slang` say "a byte an axis", but the code stores two
bytes.

### 5.7 Geometry layer

Source: `shaders/athenea/scene/video_geometry.slang`, `video_planes_common.slang` ("THE GEOMETRY LAYER"), `Planes.h` (`geometryChunkBytes`), `modules/video/src/Athv.cpp` (`AthvReader::read`).

With flags bit 4, position, scale and rotation do not travel in planes. They
travel exactly, in the record's geometry payload. Each gaussian `j` (its index
in the frame, `firstSplat + i`) has a state of four u32 words:

| Word | Content |
|---|---|
| 0 | `px \| py << 16` |
| 1 | `pz \| lsx << 16` |
| 2 | `lsy \| lsz << 16` |
| 3 | rotation word (the store's smallest-three layout) |

The words are formed as follows:

```
pa  = quantise(p_a,  lo_a,     hi_a,     2^P − 1)     P = geometryPositionBits, 10..16
lsa = quantise(clamp(ln s_a, −20, 20), lo_{4+a}, hi_{4+a}, 2^S − 1)    S = geometryScaleBits, 6..10
rotation: the store's word with each 10-bit component v rounded to R bits (R = geometryRotationBits, 5..10):
    d = 10 − R;  v' = (d == 0) ? v : min((v + 2^(d−1)) >> d, 2^R − 1) << d
```

The ranges are the record's: attributes 0–2 and 4–6. The decoder writes:

- the position, from `dequantise(pa, lo_a, hi_a, 2^P − 1)`;
- the log scale, from `dequantise(lsa, lo_{4+a}, hi_{4+a}, 2^S − 1)`;
- word 3, taken as the store's rotation word unchanged.

The header packs `P | S << 8 | order << 16 | R << 24` into one u32. When that
u32 is 0, the defaults are 16, 10, 1 and 10. When R reads as 0, it is 10.

#### Planes and chunks

The inflated payload is `ceil(splats / chunkSplats)` chunks. Each chunk has
16 byte planes of `chunkSplats` bytes:

```
chunkBytes = 16 · chunkSplats + (order == 3 ? (chunkSplats / 256) · 32 : 0)
byte of plane p for picture splat i:  chunk (i / chunkSplats) · chunkBytes + p · chunkSplats + (i mod chunkSplats)
```

| Plane | Byte |
|---|---|
| `2f`, `2f + 1` (f = 0..5) | low and high byte of 16-bit field `f`: px, py, pz, lsx, lsy, lsz |
| 12, 13, 14, 15 | bytes 0, 1, 2, 3 of the 32-bit rotation field |

Bytes for splats `≥ splats` are 0. A field holds either an absolute value or a
residual, as described next.

#### Keys and prediction (orders 1 and 2)

The decoder takes these parameters from the header and the record:

- `interval`: the header's `geometryInterval`;
- `stagger`: flags bit 5;
- `order`: the header's `geometryOrder`;
- `frame`: the record's `source` (from 0).

```
geometryKey(j)   = interval ≤ 1                                  → true
                   stagger    → (j / 256) mod interval == frame mod interval
                   otherwise  → frame mod interval == 0
order(j)         = 0  if geometryKey(j, frame)
                   1  if frame == 0 or geometryKey(j, frame − 1)
                   min(order, 2) otherwise
```

When `order(j) = 0`, the fields are the state, absolute. Otherwise, with `b`
the state of the frame before and `c` the state of the frame before that:

```
16-bit field f:  pred = (order(j) ≥ 2) ? (2·b_f − c_f) & 0xFFFF : b_f
                 sent = zigzag16((now_f − pred) mod 2^16)
                 now_f = (pred + unzigzag16(sent)) & 0xFFFF
rotation:        pred = order(j) ≥ 2 ? (b & 3) | each 10-bit component (2·b_k − c_k) & 0x3FF : b
                 sent = ((now ^ pred) & 3) << 30
                        | for k = 0..2: zigzagR(((now >> s_k) − (pred >> s_k)) mod 2^R) << 10k,
                          s_k = 22 − 10k + (10 − R)
                 decode: index = (pred ^ (sent >> 30)) & 3;
                         component k = ((pred >> s_k) + unzigzagR((sent >> 10k) & (2^R − 1))) mod 2^R, at s_k
zigzag_n(d): d as an n-bit signed s → ((s << 1) ^ (s >> 31)) & (2^n − 1);   unzigzag_n(z) = ((z >> 1) ^ −(z & 1)) & (2^n − 1)
```

After each frame, the decoder keeps both `b` (moved down a place) and the new
state. Order 2 needs the frame before that.

**The record's `geometryKey` flag** (prefix flags bit 1) is set when
`interval ≤ 1`, or when there is no `stagger` and `frame mod interval == 0`.
With `stagger` and `interval > 1`, no record has it, even though some blocks
of 256 are keys in every frame.

**Conditional replenishment** (`--geometry-hold`, orders 1 and 3) is an
encoder choice: a field within its threshold is sent as residual 0. The
decoder does nothing different.

#### Motion (order 3)

Source: `video_geometry.slang` (`videoGeometryFit`, `motionPosition`, `motionSteps`, `motionApply`, `videoGeometryApply`).

Gaussians form clusters of 256 consecutive frame indices. The cluster of
gaussian `j` is `j / 256`, and within a chunk it is `(i mod chunkSplats) / 256`.
Each chunk's 16 planes are followed by the motion of its
`chunkSplats / 256` clusters: 8 u32 words (32 bytes) per cluster.

| Word | Content |
|---|---|
| 0–2 | `c`: the held centroid x, y, z, int32, in quarter position steps (4 × the P-bit units) |
| 3–5 | `t`: the frame's centroid x, y, z, int32, quarter position steps |
| 6 | `qx` (int16, low half) `\| qy << 16` (int16) |
| 7 | `qz` (int16, low half) `\| qw << 16` (int16); 8192 = 1, `w ≥ 0` |

When `geometryKey(j)` is true, the four fields are the state, absolute. This
is the value a decoder would hold, not the source re-quantised. Otherwise, let
`h` be the held state, and let the sent fields be zigzagged as follows:

- position and log scale: 16-bit signed, as `zigzag16`;
- rotation: three 10-bit signed integers `n0 | n1 << 10 | n2 << 20`, unzigzagged as `int(z >> 1) ^ −int(z & 1)`.

The decoder then computes:

```
position:  d = 4·h_p − c                                   (int32, h_p the held P-bit position)
           if any |d_a| ≥ 2^16:  p4 = 4·h_p + (t − c)
           else, with (x, y, z, w) = (qx, qy, qz, qw):
             R = [[8192 − ((y·y + z·z) >> 12), (x·y − z·w) >> 12,  (x·z + y·w) >> 12],
                  [(x·y + z·w) >> 12,  8192 − ((x·x + z·z) >> 12), (y·z − x·w) >> 12],
                  [(x·z − y·w) >> 12,  (y·z + x·w) >> 12,  8192 − ((x·x + y·y) >> 12)]]
             p4 = t + ((R·d + 4096) >> 13)
           p^ = clamp((p4 + 2) >> 2, 0, 2^P − 1)
           p_a = (p^_a + unzigzag16(sent_a)) & 0xFFFF
log scale: ls_a = (h_ls_a + unzigzag16(sent_{3+a})) & 0xFFFF           (no prediction)
rotation:  sigma_a = exp(dequantise(ls_a, lo_{4+a}, hi_{4+a}, 2^S − 1))
           posStep = ((hi_0 − lo_0) + (hi_1 − lo_1) + (hi_2 − lo_2)) / (3 · (2^P − 1))
           base    = 4 · 0.70710678 / (2^min(R,10) − 1)
           step_a  = max(base, 0.5 · posStep / max(|sigma_{a+1} − sigma_{a+2}|, 1e-12)),   0 if > 3.14159265   (indices mod 3)
           q_R = normalize((qx, qy, qz, qw) / 8192)
           q   = normalize(q_R ⊗ decodeQuaternion(h_rot)) ⊗ exp(n ⊙ step)
                 exp(v) = (v/|v| · sin(|v|/2), cos(|v|/2)), identity when |v| < 1e-12;  ⊗ the Hamilton product (x, y, z, w)
           state rotation word = encodeQuaternion(q)
```

The ranges used are the record's floats 0–5 and 8–13 (attributes 0–2 and 4–6).
The rotation step is float arithmetic. The encoder runs this same compiled
kernel to obtain what a decoder holds, so another implementation matches it
only to the extent its float results do.

#### What the code leaves undetermined in the layer

- **Starting state.** At frame 0, a gaussian that is not a key (stagger, or
  order 3) is predicted from a state the encoder assumes is the loop's last
  frame. The encoder primes it by encoding the loop's tail, or for order 3 the
  whole loop twice. A decoder that starts cold has no defined state for those
  gaussians until their first key. The decoder's state buffer is created
  without initial contents.
- **Unused cluster records.** Order 3 cluster records for clusters with no
  gaussians in the picture are never written by `videoGeometryFit`. Their 32
  bytes are whatever the encoder's buffer held.

### 5.8 Geometry in the video

Source: `video_planes_common.slang` ("GEOMETRY IN THE VIDEO"), `video_planes_encode.slang` (`sampleOf`, `valueOf`), `video_planes_decode.slang` (`geometryAt`, `videoPlanesDecode`).

With flags bit 7 (8 bits only, no geometry layer, no high parts), position,
log scale and rotation are luma planes. Only the ranges travel beside the
video.

`geometryFold` (header offset 84) holds three folds `K`:

| Bits | Fold of |
|---|---|
| 0–7 | position (attributes 0–2) |
| 8–15 | log scale (attributes 4–6) |
| 16–23 | rotation (attributes 7–9) |

`K = 0` means coarse planes only. The nine geometry attributes, `g = 0..8`,
are attributes 0, 1, 2, 4, 5, 6, 7, 8, 9.

**Canonical rotation.** The rotation is the store's quaternion, normalised
and then made canonical. Of `q`, `q·i = (w, z, −y, −x)`, `q·j = (−z, w, x, −y)`
and `q·k = (y, −x, w, −z)`, the one with the largest `|w|` is chosen; ties
keep the earlier one in that order. It is negated if its `w < 0`. Attributes
7–9 are its x, y, z. Attribute 10's range is written but not used.

| Planes | Content |
|---|---|
| 0–8 | coarse plane of geometry attribute `g` |
| 9 .. 9 + 2F − 1 | fine planes: for each folded attribute in `g` order, wave A then wave B |
| 9 + 2F | opacity |
| after | colour planes (plain or `chroma`, as in plain mode) |

`F` is the number of folded attributes: 3 for each non-zero fold. For an
attribute with range `[lo, hi]`:

```
K = 0:   coarse = quantise(v, lo, hi, 255);           v = dequantise(coarse, lo, hi, 255)
K > 0:   N = floor(65025 / K)
         U = quantise(v, lo, hi, N)
         coarse = round(U · 255 / N)
         A = wave(U),  B = wave(U + 128),  wave(u) = m ≤ 255 ? m : 510 − m,  m = u mod 510
decode:  useA  = 64 ≤ A ≤ 191
         shift = useA ? 0 : 128;   f = min(useA ? A : B, 255)
         e     = coarse · N / 255 + shift                                 (float)
         j0    = floor(e / 510)
         U'    = the candidate among  j·510 + f  and  j·510 + 510 − f,  j = j0−1 .. j0+1,
                 nearest to e (the first found wins a tie, in the order j ascending, a before b)
         U     = clamp(U' − shift, 0, N);   v = dequantise(U, lo, hi, N)
```

The rotation's w is rebuilt as `sqrt(max(0, 1 − x² − y² − z²))`. The decoder
normalises `(x, y, z, w)` (with a floor of `1e-8` on the length) and stores it
smallest-three.

### 5.9 Volumes

Source: `modules/video/include/athenea/video/Athv.h` (`volume`), `shaders/athenea/technique/volume_splat_ply.slang`, `shaders/athenea/volume/vdb_splats.slang` (`vsImportSplats`), `shaders/athenea/scene/splat_decode.slang`.

With flags bit 8 (`video-encode --volume S`), the splats are a medium's
gaussians of extinction, read from a 3DGS PLY written as `volume_splat_ply`
lays it out. The header's `volumeDensityScale` (offset 88) is `S`: the
extinction per unit of length for a sigma of 1. Nothing else in the picture
layout changes. The quantities ride in ordinary attributes.

The PLY's colour-major `f_rest_0` and `f_rest_1` are red, basis 0 and basis 1.
The loader writes them to the store's `k = 0` and `k = 3`.

| Attribute | Content |
|---|---|
| 14 (`k = 0`, red, first basis) | ln sigma, the extinction's peak in density units; at least −9.5 for a live gaussian (floor −10 + 0.5) |
| 17 (`k = 3`, red, second basis) | the temperature |
| 11–13 | the radiance towards the eye, display-referred: `srgb(x / (1 + x))` per channel, x = exposure × L |
| 3 | for a plain viewer: `clamp(1 − exp(−sigma · S · √(2π) · d), 1/254, 0.999)`, d = exp(mean of the three log scales) |
| other rest coefficients | 0 |

A player draws each gaussian with alpha `1 − exp(−τ)`, where τ is its optical
depth along the ray: sigma × `S` per unit length.

- **Dead slots.** A dead slot (a speck) is at the origin with ln sigma −10,
  log scales `ln 1e-6`, identity rotation and opacity logit −20. The importer
  treats any splat with `ln sigma ≤ −9.75` as dead.
- **Required coefficients.** The temperature needs `restPerColour ≥ 3`
  (`--sh-degree` ≥ 1). The importer reads ln sigma when
  `restPerColour ≥ 1` and the temperature when `restPerColour ≥ 2`.
- **Undetermined.** The code does not say how a reader should treat a volume
  file with `restPerColour = 0`. It also does not say how to treat a volume
  file with `chroma`. In that case, the red coefficients of basis 0 and
  basis 1 are rebuilt from YCoCg, whose Co and Cg are shared by four splats.
  `video-encode` does not reject either combination.

### 5.10 Version 1

A 64-byte header: offsets 0 to 47 as version 2's, with offset 36 unused
(read as rendition 0) and no fraction, source frames, quality, blocks or
geometry; `positionHighBits` equals `bits`. Its record starts with 48 bytes:
magic, bytes (u32), index, splats, flags (u32, bit 0 a key), source,
firstSplat, sourceSplats (u32 each), ptpTime (u64), time (f64); then the
time code label and the same six u32 counts and data as version 2, the
geometry count ignored. A version 1 record's frame is its source; its slices
are `⌈sourceSplats / splats⌉` and its slice `firstSplat / splats`.

### 5.11 As a stream

`scripts/examples/athv_stream.py` sends one `.athv` per rendition as
messages, each a u32 length (little endian) and that many bytes. All three
kinds share the 28-byte prefix of §5.2.

| Magic | Type | Rendition | After the prefix |
|---|---|---|---|
| `ATVF` | 0 | the file's | the record, byte for byte, but for `frame` (counts on across loops; the same for the same instant in every rendition) and `ptpTime` (`CLOCK_REALTIME` when sent) |
| `ATVM` | 1 | 0xFFFF | a u32 length and that many bytes of JSON: the manifest |
| `ATVA` | 2 | 0xFFFF | a u32 length and JSON: ancillary data |

A manifest and an ancillary message carry `bytes` = their own length,
version 2 and flags 0. The manifest is sent first and then once a second,
before that frame's pictures; ancillary data right after each manifest and
again when it changes. Every rendition's pictures of frame k go before any
of frame k + 1.

**The manifest** is `{"version": 1, "fps": {"num", "den"}, "frames",
"renditions": [...]}`, each rendition the header's fields by name (`id`,
`bits`, `restPerColour`, `chunkSplats`, `columns`, `maxSplats`,
`hiInVideo`, `antialiased`, `positionBlock`, `positionHighBits`,
`chromaPlanes`, `stableOrder`, `geometryLossless`, `geometryKeyInterval`,
`geometryStagger`, `geometryPositionBits`, `geometryScaleBits`,
`geometryOrder`, `geometryRotationBits`, `geometryVideo`, `geometryFold`,
`quality`) and what a player sizes itself by: `gaussians` (the median a
frame), `maxGaussians`, `shDegree`, `picturesPerFrame`, `pictureWidth`,
`pictureHeight`, `meanBytesPerFrame`, `meanMbps`, `fullRange` (true), and,
from the first key picture, `codec` (`hvc1.…`), `codecAnnexB` (`hev1.…`),
`profile`, `tier`, `level` and `hvcC` (base64). The fields are
`scripts/examples/athv_read.py`'s `describe`.

**Ancillary data** is any JSON object; the environment block a player
lights by is `{"environment": {"url", "mime", "format": "latlong",
"toWorld": 3 × 3 rows, "intensity", "exposure", "background"}}`.

## 6. Related files

**A pages directory** (`athenea flatten --range`, read by `athenea stream`
and `athenea video-encode`): one SPZ file a page, `f<frame>_c<page>.spz`
(a `.ply` where there is no `.spz`), and `stream.txt`, a text file of
lines; `#` starts a comment.

```
rate <time codes a second>
chunks <the most pages a frame>
chunkSplats <splats a page>
degree 3
frame <index> <time code> <pages> <gaussians> <bytes>     (one a frame)
```

A frame of fewer pages than `chunks` removes the rest from the stream.

**The web's formats.** sparkwebGPU, athenea's web player, writes a `.athc`
version 3 (`ATH3`: sections, blocks and pages, each gzipped or not, as
words, byte planes or byte-plane differences, with a CPCA transfer) and a
web scene `.athz` (a zip holding `scene.json` and its clouds). Both are
sparkwebGPU's formats and are specified in its repository; they share
nothing with athenea's `.athz` (§4) but the name.

**Live pages** (`ATVP`, `modules/lod/include/athenea/lod/DevicePages.h`) and
staged SPZ pages (`LSPZ`) are in-process headers on bytes a producer hands
the streaming pool; no file holds them.
