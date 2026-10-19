<!-- Copyright (c) 2026 jesus luque. -->

# 3. From USD meshes to Gaussians (mesh2splat)

The conversion is the relight baker's first stage: it turns a USD stage of polygonal meshes and their materials
into a cloud of anisotropic 3D Gaussians, and that cloud is what the transfer bake of Section 4 fills with light. It
is invoked as `athenea mesh2splat STAGE`; every Gaussian the rest of the system draws, bakes, decimates and streams
was a triangle of a mesh first. This section
describes how the stage is read, how large a Gaussian's cell is, how each Gaussian is placed, shaped and coloured,
how the conversion handles the places where a surface stops or folds, how it captures a layered physically based
material, how it treats glass, and what was measured on the Chevrolet Corvette C7 asset that drives most of the
project. The path-traced bake that follows emission (the *transfer*) is the subject of Section 4 and is
touched on here only where it constrains the conversion.

::: {.nota}
**Mesh → Gaussian conversion** — A USD polygonal mesh (`UsdGeomMesh`) is a list of points, a face topology (`faceVertexCounts`, `faceVertexIndices`) and a set of primvars (normals, texture coordinates, colours) with a declared interpolation (`vertex`, `faceVarying`, `uniform`, `constant`). An anisotropic 3D Gaussian is a volumetric primitive with centre $\boldsymbol\mu$, covariance $\Sigma$ (a symmetric positive definite $3\times3$ matrix), opacity $\alpha$ and colour (a DC term and, optionally, spherical harmonics). Its unnormalised density is

$$
G(\mathbf x) = \exp\!\left(-\tfrac12(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\right),
$$

and the covariance is stored factored as $\Sigma = R\,S\,S^\top R^\top$, with $R$ a rotation (a quaternion) and $S=\operatorname{diag}(s_1,s_2,s_3)$ three standard deviations, exactly as in 3D Gaussian Splatting [1]. The fundamental difference from [1] is where the cloud comes from: in 3DGS the Gaussians are optimised against photographs (inverse rendering, with adaptive densification and pruning); here they are built deterministically from a geometry known exactly, with no photometric loss and no gradient descent. The cloud's error therefore does not come from a fit, but from what a Gaussian cannot represent: a single normal, a finite footprint, no refraction.

**Relight baker** — The final product is not a capture with its light frozen in, but a cloud whose records store how each Gaussian responds to illumination (the *transfer* of §4, in the line of Sloan et al.'s precomputed radiance transfer [14]). The conversion fixes the geometry and the material; the bake fixes the response to light.

**USD stage** — A stage is the composed view of a set of layers (`.usda`, `.usdc`, references, payloads, variants) [43]. "Decimates" and "streams" refer to the merge trees, cuts and pages of §6; "draws" to the rasteriser and the ray tracer of §2.

*References:* [1], [12], [14], [43]; code: `apps/athenea/src/CmdMesh2Splat.cpp` (class `Converter`, which orchestrates the conversion); manual: `docs/operations.md` §2.8 ("`athenea mesh2splat` — a model into gaussians") and §3.1; `docs/decisions.md`, section "mesh2splat: a mesh becomes a cloud, through the plugin interface".
:::

Numbers in this section come from `docs/decisions.md` of the repository and from the source (`plugins/mesh2splat/mesh2splat.slang`,
`apps/athenea/src/CmdMesh2Splat.cpp`, `shaders/athenea/usd/mesh2splat_*.slang`). All measurements were taken on an
Apple M5 Pro (Metal) unless said otherwise.

::: {.nota}
**Where each part lives** — The split across files reflects the repository's rule that the processor does no arithmetic on data (the repository's `CLAUDE.md`, "No CPU arithmetic on data"):

- `plugins/mesh2splat/mesh2splat.slang` is the kernel derived from Electronic Arts' (its BSD-3 licence is kept verbatim in the file's header) with the changes this section describes: entry points `m2sClear`, `m2sCount`, `m2sScan` and `m2sEmit`, and the shared function `m2sTriangleOf`.
- `plugins/mesh2splat/Mesh2Splat.cpp` is the host side of the AOFX effect (class `Mesh2Splat final : public aofx::Effect`): it declares parameters and inputs, reserves the working buffers (`scratch`) and launches the three passes.
- `apps/athenea/src/CmdMesh2Splat.cpp` is the command: it reads the stage, splits it into pieces, decides the cells, shares the budget, launches the effect piece by piece, bakes and writes the output.
- `shaders/athenea/usd/mesh2splat_*.slang` are the kernels the command runs around the effect: `mesh2splat_cells` (cell per piece), `mesh2splat_edges` (hard edges), `mesh2splat_sheet` (glass classification), `mesh2splat_subset` (a GeomSubset's triangles), `mesh2splat_gather` (records into the cloud's buffers), `mesh2splat_span` (box and ray offset for the bake) and `mesh2splat_bake` (the bake's response into the records).

**Apple M5 Pro (Metal)** — The time and memory figures come from a single machine with unified memory; the device memory limits cited in §3.11 (a tree of 5.01 M leaves that does not fit) depend on it.

*References:* `docs/development.md` (the section "An aofx plugin"); `docs/decisions.md`, "mesh2splat's host side: on the device, by subset, and fair to every mesh".
:::

## 3.1 Origin, and where the conversion runs

The emission algorithm is that of Electronic Arts' *mesh2splat* (BSD-3): every triangle is projected onto the axis
plane its normal points along most, the projection is rasterised at a fixed resolution, and each covered texel
becomes one Gaussian lying in the triangle's plane, as wide as a texel measured in the world. The original is an
OpenGL geometry and fragment shader pair; here it is one compute kernel in which each thread owns a triangle and
walks the cells of its projected bounding box.

::: {.nota}
**Electronic Arts' mesh2splat (SEED)** [12] — The original is an OpenGL pipeline: a geometry shader receives each triangle, picks the axis its normal points along most, projects the triangle onto the coordinate plane perpendicular to that axis and computes a Jacobian from those coordinates to world space; a fragment shader runs on every covered texel, samples the maps and appends a Gaussian to a buffer with an atomic counter.

**Projection onto the dominant-axis plane** — This is the same choice made by triplanar mapping and by rasteriser-based voxelisation (Crassin and Green, "Octree-Based Sparse Voxelization Using the GPU Hardware Rasterizer", *OpenGL Insights*, 2012): of the three coordinate planes, the one onto which the triangle projects the largest area is taken. If $\hat{\mathbf n}$ is the unit normal and $k$ the axis with the largest $|n_k|$, the projected area is $A\,|n_k|$ with $|n_k| \ge 1/\sqrt3$, so no triangle is seen at a cosine worse than $0.577$ (about $54.7^\circ$). A regular grid in the projected plane induces on the triangle a spacing between samples that, along the direction of steepest slope, is $h/|n_k| \le \sqrt3\,h$; the Jacobian of §3.4 compensates for that stretching by elongating the Gaussian in that direction.

**"Rasterised at a fixed resolution"** — The grid has $N$ cells (`--resolution`) across the reference range $R$; the original's texel is the cell here.

**One thread per triangle** — With no fixed-function rasteriser, each thread walks the cells of its triangle's projected bounding box and tests whether the centre of each one falls inside (`m2sCellInside`, the pixel-centre rule, the same one a rasteriser applies). The sample set is identical to the original's at the same density. Advantages: a single portable kernel (Slang to Metal and CUDA), explicit control of the output order (§3.4, "Deterministic order") and no dependence on the graphics pipeline. Drawback: load imbalance, because a large triangle keeps one thread busy for thousands of cells while its small neighbours finish at once; hence the `--max-cells` cap of §3.3 and the block walk `m2sWalkBlocks` (option `--simplify`).

*References:* [12]; code: `plugins/mesh2splat/mesh2splat.slang` (header comment "WHAT THEIR PIPELINE DOES, AND WHAT THIS ONE DOES INSTEAD", functions `projectionAxes`, `orthogonalUv`, `m2sTriangleOf`, `m2sCellInside`).
:::

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

::: {.nota}
**AOFX effect** — AOFX is aopenfx's effect plug-in interface (`third_party/aopenfx`): an effect declares inputs and parameters (`describe`) and receives device images with which it runs kernels; the host registry and runner are shared by every program that loads bundles (§2.14). Making the conversion an effect has two consequences: the converter stays outside the renderer's modules (it does not link against `athenea::render` or `athenea::technique`), and the same bundle can run in any AOFX host (a compositor, for instance). The AOFX ABIs change only additively; that is why every new parameter of the effect (`cellMin`, `cellMax`, `cellByLongest`, `edgeReach`, `normalSpread`…) has a default value that reproduces the previous behaviour.

**The picture of numbers** — An effect receives images, not meshes, so each triangle travels as six `float4` pixels: for each of its three corners, $(\mathbf p, u)$ and $(\mathbf n, v)$ in world space, 96 bytes per triangle. The `meshPack` kernel of `mesh_pack.slang` writes it: one thread per triangle, positions read from the mesh's point buffer with the indices `geom::MeshBuilder` triangulates, the world transform applied in the kernel, and normals and UVs resolved according to their interpolation, as `world/primvars.slang` does in the renderer.

**Textures as rows of `float4`** — An effect sees no samplers and no mips. `texture_rows.slang` (`textureRows`, one thread per texel) samples the texture through the same texture table every material of the engine uses, so colour space, wrap and filtering match the renderer's; the result is linear and in bottom-up row order ($v$ upwards). A texel in `float4` takes 16 bytes, hence the `--texture-size` limit of §3.2.

**No copy (gpe)** — gpe is the image storage layer that adopts slang-rhi's device (a single `MTLDevice`, or a CUDA context). `Context::renderView` returns a `gpu::Buffer` that is a view over the same bytes as a gpe image, so the packing kernel writes into the memory the effect then reads, with no transfer.

**Gather and span** — `mesh2splat_gather` (`m2sGather`) places each run's records into the cloud's buffers at their offset; `mesh2splat_span` reduces the cloud's box on the device (`m2sRecordChunks` and `scene/bounds_reduce`) and writes into every bake ray an offset of $10^{-4}$ of that box's diagonal (`m2sRaySpan`), a fraction of the model rather than of the scene unit, so that the ray does not start inside the triangle it is about to meet.

**"Only counters and the final USD arrays cross to the processor"** — This is the repository's rule: the processor reads files and keeps counts; everything else is done by kernels. The USD arrays cross because the USD API requires arrays in host memory to write attributes.

*References:* §2.14 of the paper; code: `plugins/mesh2splat/Mesh2Splat.cpp`, `modules/aofx/host/src/Host.cpp`, `shaders/athenea/usd/mesh_pack.slang`, `shaders/athenea/usd/texture_rows.slang`, `modules/gpu_host/include/athenea/gpu_host/Context.h` (`renderView`), `modules/gpu_host/include/athenea/gpu_host/Views.h`, `shaders/athenea/usd/mesh2splat_gather.slang`, `shaders/athenea/usd/mesh2splat_span.slang`; `docs/decisions.md`, "One AOFX host, from aopenfx" and "mesh2splat's host side: on the device, by subset, and fair to every mesh".
:::

The command is therefore a pipeline of device passes orchestrated by the host (Figure 6):

1. read the stage (`usd::MeshStage`) and split meshes into *pieces*;
2. pack each piece into a mesh picture, fold boxes, choose each piece's cell (`mesh2splat_cells`);
3. mark hard edges and closed shells per mesh (`mesh2splat_edges`), classify glass (`mesh2splat_sheet`);
4. run the Mesh2Splat effect per piece — count, scan, emit — into records;
5. optionally bake (radiance, or a transfer) at every Gaussian with the path tracer;
6. write a `ParticleField3DGaussianSplat` stage (or a `.athc`) atomically.

::: {.nota}
**The six stages** — (1) `usd::MeshStage::read` returns the stage's `StageMesh`es and the command splits them into pieces (`Converter::makePieces`). (2) `Converter::packMeshes` packs each piece with `meshPack`, folds the boxes by chunks (`streamBoundsChunks` and `bounds_reduce_slices`) and `Converter::deriveCells` runs `m2sCells`. (3) `Converter::markEdges` (hard edges and closed shells, §3.5) and `Converter::classifySheets` (glass, §3.8). (4) The effect, in three passes: count the covered cells per triangle, the exclusive prefix sum and the emission (§3.4). (5) The bake (§4), optional: baked radiance or transfer. (6) The write, atomically: `platform::writeAtomically` writes to a temporary file `.<name>.partial-<pid>.<ext>` in the same directory and renames it when done, so an interrupted conversion leaves no partial cloud under the name the next step reads.

**`ParticleField3DGaussianSplat`** — OpenUSD schema (the `ParticleField` family, from USD 26.x on) describing a cloud of Gaussians: positions, orientations, scales, opacities and radiance coefficients; athenea's own channels go as `primvars:athenea:splat:*` primvars. `.athc` is the player's compressed format (§7).

**Correspondence with Figure 6** — The figure breaks the list into nine boxes: stage 2 of the list is its box 2, stage 3 its box 3, stage 4 is spread over its boxes 4 to 7 (emission, edge pass, material capture and specular filter, which in the code happen inside `m2sWrite`), stage 5 is its box 8 and stage 6 its box 9.

*References:* code: `apps/athenea/src/CmdMesh2Splat.cpp` (`makePieces`, `packMeshes`, `deriveCells`, `markEdges`, `classifySheets`), `modules/core/include/athenea/core/Platform.h` (`writeAtomically`), `modules/usd/src/Export.cpp`; `docs/operations.md` §8.2 (formats); `docs/decisions.md`, "mesh2splat's host side…" (subsection "Whole or not at all").
:::

![Figure 6. The conversion as a sequence of device passes. Steps 2–7 are kernels or the Mesh2Splat AOFX effect; step 8 is the transfer bake of Section 4.](figures/fig-mesh2splat.svg)

## 3.2 Reading the stage

`usd::MeshStage` walks the stage with `UsdShade` directly rather than through Hydra: the conversion wants no render
index, and the material side is built on the stage's own bindings.

::: {.nota}
**UsdShade versus Hydra** — Hydra is OpenUSD's rendering architecture: a scene delegate (UsdImaging) translates the stage into a render index of render prims (Rprims, Sprims, Bprims) that a render delegate consumes. It is the renderer's route (the `hdAthenea` delegate, §2). The conversion needs no render index, only meshes, transforms and materials, so `MeshStage` walks the stage directly and resolves materials with the UsdShade API (`UsdShadeMaterialBindingAPI::ComputeBoundMaterial`, which applies inheritance, binding purposes and GeomSubsets).

**The risk of having two readers** — The renderer (through Hydra) and the conversion (through `MeshStage`) must agree on what is visible, how it is instanced and which material each face carries; when they disagree, the cloud is not what the renderer draws. The following paragraphs (visibility, instancing, GeomSubsets) are largely the points where they were brought into line.

*References:* [43]; code: `modules/usd/src/MeshStage.cpp` (`MeshStage::read`), `modules/usd/include/athenea/usd/MeshStage.h`; `docs/decisions.md`, "mesh2splat converts what the renderer draws" and "An invisible PointInstancer hides its instances".
:::

**Selection.** `--prim P` restricts conversion to meshes at or under `P`; `--hide P` (repeatable) authors
invisibility on `P` and everything below it, in a non-persistent layer over the stage, before the read. A mesh that computes invisible at the read's time code
is skipped, so the conversion converts what the renderer draws. `--hide` affects the conversion only: the bake
traces the whole stage, so a part converted with every other part hidden still receives the others' shadows and
inter-reflections. This is what allows the Corvette to be converted in groups (Section 3.11).

::: {.nota}
**`--prim` and `--hide`** — `--prim P` is a traversal filter. `--hide P` writes `visibility = invisible` on `P` in the stage's *session layer*, the strongest, non-persistent layer that USD keeps in memory and never saves alongside the file (`UsdEditContext` on `GetSessionLayer()`). USD visibility is inherited: an invisible prim makes its whole subtree invisible, and the effective visibility is computed at a time code (`ComputeVisibility`), which explains the phrase "computes invisible at the read's time code".

**Why the bake does not honour `--hide`** — The bake traces the complete original stage. A part converted alone still receives the shadow and inter-reflections of the others, which allows a large asset to be split into independent conversion groups (device memory, time per group) without each group's lighting depending on the split. It is the basis of the eight groups at 1280 px and the twenty-four trees at 5120 px of §3.11.

*References:* code: `modules/usd/src/MeshStage.cpp` (the `--hide` block inside `MeshStage::read`, which returns "no prim that can be seen there" if the path does not exist), `apps/athenea/src/CmdMesh2Splat.cpp` (option `--hide`); `docs/operations.md` §2.8.
:::

**Instancing.** A `PointInstancer` prototype's mesh is emitted once per instance of the innermost instancer that
holds it, at (mesh relative to its prototype root) × (instance transform, prototype root included) × (instancer to
world); the built mesh buffers are shared between copies. Without this expansion the chess set's sixteen pawns
convert to two.
Nested instancers are expanded at the innermost level only.

::: {.nota}
**`PointInstancer`** — A USD prim that draws many copies of a set of prototypes: `prototypes` (relationship to the prototype subtrees), `protoIndices` (which prototype each instance uses) and `positions`, `orientations`, `scales` (each instance's transform). `UsdGeomPointInstancer::ComputeInstanceTransformsAtTime` returns one matrix per instance; with the `IncludeProtoXform` option, the prototype root's own transform is included.

**Composing the transforms** — USD uses row vectors ($\mathbf v' = \mathbf v\,M$), so the matrix reads from left to right in the order it is applied:

$$
M_{\text{world}} = M_{\text{mesh}\to\text{prototype root}}\; M_{\text{instance}}\; M_{\text{instancer}\to\text{world}}.
$$

The mesh is taken relative to its prototype root precisely because that root's transform is already inside $M_{\text{instance}}$; taking it in world space would apply it twice.

**Shared buffers** — The triangulation, normals and expanded primvars (`geom::MeshBuilder`) are computed once per prototype; only the matrix changes when each copy is packed.

**The pawns** — In the test chess set the sixteen pawns are instances of two prototypes (white and black); without the expansion, the conversion read only the prototypes at their original position.

**Nesting** — An instancer inside another's prototype would produce the product of both sets of instances; here only the innermost is expanded (a limitation repeated in §3.12).

*References:* [43]; code: `modules/usd/src/MeshStage.cpp` (structure `InstancedAt`, function `instancesOf`); `docs/decisions.md`, "mesh2splat converts what the renderer draws".
:::

**Time and pose.** `--time T` sets the time code at which positions, transforms and primvars (including their
indices) are read and at which the bake traces. A skinned mesh's `points` do not animate in USD, so the stage is
first posed with `UsdSkelBakeSkinning` into a non-persistent layer over $[T, T]$, which costs one pose and leaves the file
untouched; a stage without a `SkelRoot` is unchanged and converts byte-identically. With `--skinned` the Gaussians
are built in the bind pose and each keeps four joints and their weight gradients; that path is described in
Section 5.

::: {.nota}
**`--time T`** — Sets the `UsdTimeCode` at which time-sampled attributes are evaluated: positions, transforms and primvars, including their indices (an indexed primvar stores values and indices separately, and both may vary over time). The bake traces at the same instant.

**Skinning in USD (UsdSkel)** — A skinned mesh's `points` hold the bind pose; the deformation is described by a `Skeleton` animated by a `SkelAnimation` and applied by whoever reads it. It is linear blend skinning: with weights $w_j$ and joint matrices $M_j$ in the current pose and $B_j$ in the bind pose,

$$
\mathbf p' = \sum_j w_j\,\mathbf p\,B_j^{-1} M_j \quad (\text{row vectors}).
$$

**`UsdSkelBakeSkinning`** — A UsdSkel function that evaluates that deformation and writes the posed points as attributes; here it runs in the session layer over the interval $[T, T]$, which costs exactly one pose and does not modify the file (`poseStage`). A stage without a `SkelRoot` has nothing to pose and the conversion comes out byte-identical, a check that the new route does not alter the old one.

**`--skinned`** — The alternative: convert in the bind pose and store in each Gaussian four joints with their weights and the gradients of those weights along its two axes (record entries 6 to 8: `blendInfluences` and `m2sPackHalves` pack the gradients into half precision). The cloud is then deformed on the device (§5). The two routes are mutually exclusive: either the stage is posed before converting or the rig is carried along.

*References:* [43]; code: `modules/usd/src/MeshStage.cpp` (`poseStage`, `resolveSkinning`, `skinningOf`), `plugins/mesh2splat/mesh2splat.slang` (`blendInfluences`); `docs/decisions.md`, "The conversion reads a pose, not a rest" and "A gaussian carries the joints its triangle carries".
:::

**Pieces.** A mesh whose `materialBind` GeomSubsets bind different materials is converted as several *pieces*: each
subset's triangles, and those no subset claims. A piece's triangles are listed on the device in the mesh's order
(flag, prefix sum, scatter) and packed as if they were a mesh, so the effect is untouched. A test mesh with two faces
bound red and blue by subsets and the mesh bound green converts to 272 red and 272 blue Gaussians and none green;
reading the mesh's own binding alone makes it all green.

::: {.nota}
**GeomSubset and the `materialBind` family** — `UsdGeomSubset` is a child prim of a mesh that selects a subset of its faces (`elementType = face`, `indices`). Subsets are grouped into families by `familyName`; the `materialBind` family is the one USD uses to assign materials by groups of faces: each subset carries its own `material:binding`. Faces no subset claims take the mesh's own binding, which is what `ComputeBoundMaterial` answers for them.

**Piece** — The unit of conversion: a whole mesh, the triangles of one subset, or those no subset claims. Each piece carries one material, and therefore its own effect parameters.

**Listing on the device (stream compaction)** — `geom::MeshBuilder` already knows which subset each triangle is in (`triangleSubsets`: 0 none, $k+1$ the $k$-th). To extract a piece's list: (1) a kernel flags each triangle of the piece with 1 (`m2sSubsetFlags`), (2) an exclusive prefix sum of the flags gives each flagged triangle its position in the list (`gpu::PrefixSum`), (3) a kernel writes each flagged triangle at that position (`m2sSubsetScatter`). It is the standard GPU compaction pattern; the prefix sum preserves the mesh's order, so the conversion is deterministic. `mesh_pack` packs that list as if it were a mesh and the effect does not change.

**272** — This is the count a unit square gives at resolution 16: $16\times16 = 256$ cell centres, plus the 16 that fall exactly on the diagonal shared by the two triangles and that the closed inclusion test ($\lambda_k \ge 0$) counts in both.

**The Cryptomatte identifier** — Remains the mesh's: a GeomSubset is not a prim Hydra draws separately.

*References:* [43]; code: `shaders/athenea/usd/mesh2splat_subset.slang`, `modules/gpu/include/athenea/gpu/algo/PrefixSum.h`, `modules/geom/include/athenea/geom/Mesh.h` (`MeshBuilder`); test: `tests/mesh2splat/test_outputs.cpp` ("a mesh's GeomSubsets are converted each with its own material") on `tests/data/two_subsets.usda`; `docs/decisions.md`, "mesh2splat's host side…" (subsection "A mesh of several materials, converted as several").
:::

**Materials in their own words.** The bound surface — MaterialX `standard_surface`, OpenPBR `open_pbr_surface`,
`gltf_pbr` or `UsdPreviewSurface` — is read with its own input names and its own defaults (standard_surface 0.8 grey
at roughness 0.2, OpenPBR roughness 0.3, glTF a fully rough metal, UsdPreviewSurface 0.18 at 0.5), following
connections through node graphs and interface inputs to the image node that produces a value. A value computed by a
graph (a mix, a noise) has no answer without shading a point, so the default stands and a warning is printed; a TX
bake later reads such fields back from the compiled material on the device. A texture's colour space is taken from
the file input's `colorSpace` metadata (or `sourceColorSpace`), and maps are capped at `--texture-size` (1024 by
default): a map is a `float4` picture on the device, so a 4k map would occupy 268 MB.

::: {.nota}
**The four surface vocabularies** — `standard_surface` is Autodesk's model that MaterialX includes in its library [31]; `open_pbr_surface` is the Academy Software Foundation's OpenPBR Surface [30], successor to it and to Adobe Standard Material; `gltf_pbr` is glTF 2.0's metal-roughness model as MaterialX ships it; `UsdPreviewSurface` is USD's minimal interchange model. Each names its inputs in its own way and has its own defaults, which are what any conforming renderer applies when the input is not authored:

- standard_surface: `base` 0.8 × `base_color` 1 (a 0.8 grey), `specular_roughness` 0.2;
- OpenPBR: `base_weight` 1 × `base_color` 0.8, `specular_roughness` 0.3;
- glTF: `base_color` 1, `metallic` 1, `roughness` 1 (hence "a fully rough metal");
- UsdPreviewSurface: `diffuseColor` 0.18, `roughness` 0.5, `metallic` 0.

Reading a material with another vocabulary's defaults changes its appearance: a standard_surface with no authored inputs read with the generic defaults (white, roughness 0.5) came out matte white where the mesh is an almost glossy 0.8 grey.

**Following connections** — An input can be connected to the output of a nodegraph, which in turn reads an interface input of the material; the chain is followed to the image node (`image`, `tiledimage`, `UsdUVTexture`) whose file gives the value. A value produced by a graph (a mix, a procedural noise) cannot be known without evaluating the graph at a point of the surface: the conversion keeps the default and warns once (`warnOnce`); the TX bake later evaluates those fields with the material compiled to Slang on the device.

**A texture's colour space** — In MaterialX the `colorspace` attribute, and in USD the file input's `colorSpace` metadata, say which space the encoded values are in; `UsdUVTexture` additionally has `sourceColorSpace` (`auto`, `raw`, `sRGB`). An sRGB albedo map must be decoded to linear; a normal or roughness map is `raw` and is left untouched.

**268 MB** — A map of $4096\times4096$ texels in `float4` takes $4096^2 \times 16\,\text{B} = 268\,435\,456\,\text{B}$. With the default limit of 1024, a map takes 16.8 MB.

*References:* [30], [31], [43]; code: `modules/usd/src/MeshStage.cpp` (`colourDefault = preview ? 0.18F : gltf ? 1.0F : 0.8F`, the reading of `sourceColorSpace`, `warnOnce`), `apps/athenea/src/CmdMesh2Splat.cpp` (option `--texture-size`); `docs/decisions.md`, "A material is read in its own words, at its own defaults", "A texture's colour space, which a material network drops" and "A material's fields read back from the material itself (matx)".
:::

## 3.3 The size of a cell

Everything downstream scales with the cell $h$, the world length one Gaussian stands for. The conversion offers four
ways to choose it, in increasing order of preference.

::: {.nota}
**The cell $h$** — It is the spacing between Gaussian centres on the surface and, up to the factor $\sigma$, their width. The surface density is $1/h^2$ Gaussians per unit area, so every cost scales with $h^{-2}$: number of Gaussians, memory, bake time (a fixed number of paths per Gaussian) and merge tree size. Halving $h$ multiplies all of that by four where the surface is resolved by cells, and by less where there is already about one Gaussian per triangle (§3.11).

*References:* `docs/operations.md` §2.8 (options `--resolution`, `--density`, `--cell-min`, `--cell-max`, `--cell-from-camera`, `--camera-pixels`, `--max-splats`).
:::

**Per model** (`--density per-model`, the default when no camera is given). $h = R / N$, where $N$ is
`--resolution` (512 by default) and $R$ the wider of the projection plane's two ranges of the model's box — the
original's `orthogonalUvs`. Two defects followed: a 16-unit floor diluted the car standing on it, and a panel
facing forward walked a cell 2.5 times finer than the hood, because its projection plane is the car's short side.

::: {.nota}
**Per model** — $R$ is the wider of the two ranges of the model's box in the triangle's projection plane; it is the original's `orthogonalUvs`, the side of the unit square to which the projected coordinates are normalised.

**First defect: the floor dilutes the car** — With a 16-unit floor in the stage, $R = 16$ and $h = 16/512 = 0.03125$; the car alone, in the same model, would have a cell of about 0.0037. The floor, which is what defines the box, takes $512^2 = 262\,144$ Gaussians of the 375 199 in total.

**Second defect: the cell depends on orientation** — On a car 0.77 wide, 1.92 long and 0.60 high (model units), a panel facing forward projects onto the width × height plane, whose wider range is 0.77; the hood projects onto width × length, whose wider range is 1.92. The cells are in the ratio $1.92/0.77 \approx 2.5$.

*References:* code: `plugins/mesh2splat/mesh2splat.slang` (`m2sTriangleOf`, computation of `range`; the comment quotes the same 2.5); `docs/decisions.md`, "The density is a mesh's own, not the stage's".
:::

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

::: {.nota}
**Per mesh** — $h = L_m/N$ with $L_m$ the longest side of the mesh's box. `cellByLongest` makes every triangle of the mesh use that side as its range, whatever its projection plane: a mesh has a single cell, which removes the second defect of the per-model mode.

**The derived bounds** — If not given, $h_{\max} = h_M$ (no mesh ends up coarser than in the per-model mode) and $h_{\min} = h_M/8$ (a screw does not walk a grid a thousand times finer than the body). The density ratio between the finest and coarsest part is $(h_{\max}/h_{\min})^2 = 8^2 = 64$.

**Reading the table** — With the defaults the bumper is bounded from below: its own cell would be 0.0014 and is raised to $h_{\min} = 0.03125/8 = 0.0039$; it goes from 762 to 3 082 Gaussians and the hood from 2 326 to 20 279, while the floor keeps exactly its 262 144 because its own cell is already the model's. The control row fixes $h_{\min} = h_{\max} = 0.03125$: every mesh walks the model's cell and the total matches the per-model mode to 0.01 % (375 157 against 375 199). What remains (764 against 762, 2 339 against 2 326) is the effect of measuring against the longest side instead of the projection plane's range.

**Implementation** — Each mesh's box is folded on the device from the per-chunk boxes `streamBoundsChunks` already writes; `m2sCells` derives the bounds and each piece's cell. In the effect, the cell is bounded in world units and `perCell` is recomputed only where a bound acts, so a run without bounds keeps $1/N$ bit for bit.

*References:* code: `shaders/athenea/usd/mesh2splat_cells.slang` (`m2sCells`: `cellMax` takes `modelCell` and `cellMin` takes `cellMax / 8.0`), `plugins/mesh2splat/mesh2splat.slang` (`cellByLongest`, block "Bounded in the world"); test: `tests/data/two_planes.usda`; `docs/decisions.md`, "The density is a mesh's own, not the stage's".
:::

**From a camera** (`--cell-from-camera CAM --camera-pixels P`, after Mesh2GS, arXiv 2606.21898). Each piece's cell
is what one pixel of camera `CAM`, `P` pixels across, covers where the piece's mesh box is nearest to it:

$$
h = \frac{z\,(A/P)}{f}, \qquad z = \max\bigl(\lVert \operatorname{clamp}(\mathbf e, \mathbf b_{\min}, \mathbf b_{\max}) - \mathbf e \rVert,\; z_{\text{near}}\bigr),
$$

with $A$ the camera's horizontal aperture, $f$ its focal length, $\mathbf e$ the eye, and the result bounded only by
explicit `--cell-min`/`--cell-max`. Mesh2GS takes $z_{\min}$ over the box inside the frustum and over the camera's
time samples; the Euclidean distance at `--time` is the same for a box in view and finer, never coarser, for one
beside the frustum. On two equal cards three and seven units from a 64-pixel camera, the near card received 930
Gaussians and the far one 169: a ratio of 5.5 against $(7/3)^2 = 5.4$ expected. This is the mode used for every
production bake of the Corvette (`--cell-from-camera /World/Camera --camera-pixels` 640, 1280 or up to 5120).

::: {.nota}
**The cell from a camera** — With a pinhole camera, one sensor pixel measures $A/P$ (horizontal aperture over pixels across). By similar triangles, at a distance $z$ from the optical centre that pixel covers in the world

$$
h = z\,\frac{A/P}{f}.
$$

In USD, `horizontalAperture` and `focalLength` are expressed in the same units (tenths of a scene unit by convention, in practice millimetres), so their quotient is dimensionless and $h$ comes out in world units, those of $z$. An illustrative example: with $f = 50$ mm, $A = 36$ mm, $P = 1280$ and $z = 5$ m, $h = 5 \cdot (36/1280)/50 \approx 2.8$ mm, the order of the 3–4 mm leaves on the hood at 640 px cited in §3.11.

**The distance $z$** — $\operatorname{clamp}(\mathbf e, \mathbf b_{\min}, \mathbf b_{\max})$ is the point of the box nearest to the eye $\mathbf e$ (each coordinate is clamped separately), so $z$ is the Euclidean distance from the eye to the box: zero if the eye is inside, in which case $z_{\text{near}}$, the camera's near plane, is used. Taking the nearest point is the conservative choice: at that point the piece gets one cell per pixel, and at any point further away the cell comes out finer than the pixel.

**Differences from Mesh2GS** [13] — Mesh2GS takes the box's minimum depth inside the frustum and over the camera's time samples. The Euclidean distance at the `--time` instant coincides with it for a box in view; for a box outside the frustum, the Euclidean distance is smaller than the depth it would have "inside", which gives a finer cell, never a coarser one. The third scale Mesh2GS proposes, $r_z = D_w^2/(4 z_{\min})$, was not adopted: the width of the converted Gaussian was measured separately (§3.4, "Width").

**The two-card test** — The number of Gaussians on a surface goes as $A_{\text{surf}}/h^2 \propto 1/z^2$, so two equal cards at 3 and 7 units should be in the ratio $(7/3)^2 \approx 5.44$; the measurement gives $930/169 \approx 5.50$.

**`--camera-pixels`** — Need not be the render resolution: with 640 px of conversion and a 1920 px frame, a cell covers about three screen pixels; with 5120 px, less than half of one.

*References:* [13]; code: `shaders/athenea/usd/mesh2splat_cells.slang` (comment "FROM A CAMERA" and `const float3 nearest = clamp(eye, low, high)`), `apps/athenea/src/CmdMesh2Splat.cpp` (`deriveCells`); test: `tests/mesh2splat/test_outputs.cpp` ("a camera's pixel decides each mesh's cell"); `docs/decisions.md`, "mesh2splat's host side…" (subsection "From a camera (Mesh2GS, arXiv 2606.21898)").
:::

**The budget.** `--max-splats` (2 M by default) is shared over the whole stage. Every piece is first counted (a run
of the effect with room for one Gaussian, whose count and scan cover everything a run would write); where the total
$W$ exceeds the budget $B$, piece $i$ wanting $w_i$ gets $s_i = \max(1, B w_i / W)$ and walks a cell
$\sqrt{w_i / s_i}$ times coarser. A cell $k$ times coarser covers the surface with $k^2$ fewer Gaussians, so
proportional shares and uniform coarsening are the same thing. On two cards wanting about 1 300 under a budget of
800 the shares came out 356 and 361; spent in mesh order, the second card held about 145.

::: {.nota}
**Sharing the budget** — Each piece is counted first: a run of the effect with room for a single Gaussian, whose count and prefix-sum passes already compute how many it would want to write ($w_i$). If $W = \sum_i w_i \le B$ every piece gets what it wants. Otherwise piece $i$ gets $s_i = \max(1, B\,w_i/W)$ and its cell is coarsened by the factor $f_i = \sqrt{w_i/s_i}$.

**Why proportional shares equal uniform coarsening** — A cell $f$ times coarser covers the same surface with $f^2$ times fewer Gaussians, so $w_i/f_i^2 = s_i$. And since $s_i/w_i = B/W$ for every piece (except for the minimum of 1), $f_i = \sqrt{W/B}$ is the same factor for all of them: sharing proportionally is coarsening everything equally, with no piece disappearing.

**Reading the measurement** — Two cards wanting about 1 300 in total under a budget of 800 would ideally get 400 each; they get 356 and 361. The difference is discretisation: the actual count of a grid over a triangle does not scale exactly as $1/f^2$ (borders, centres on shared edges). With the old spending in mesh order, the first card took almost everything and the second about 145.

*References:* code: `apps/athenea/src/CmdMesh2Splat.cpp` (`wantedBy_`, `shareOf_`, warning "the budget is shared in proportion"), `shaders/athenea/usd/mesh2splat_cells.slang` (entries `cells[3 + P + p]`, the factor $\sqrt{w/s}$); test: `tests/mesh2splat/test_outputs.cpp` ("a budget too small is shared between the meshes, not spent on the first"); `docs/decisions.md`, "mesh2splat's host side…" (subsection "The budget is shared, and a camera can decide the cell").
:::

**Per-triangle and per-run limits.** A triangle walks at most `--max-cells` cells of its projected bounding box
($2^{18}$ by default); cells beyond are counted and reported as a warning. At 4736 px the Corvette's
`Infinite_Black` lost 359 628 cells this way (visible holes) and was re-baked with `--max-cells 1048576`. A run's
output picture is capped at 2 M Gaussians (`kRunCeiling`, 192 MB); a piece wanting more is converted in slices — the
effect reports the first triangle the ceiling cut into, and the host reruns from there with `firstTriangle` — giving
the same array a single run would have written.

::: {.nota}
**`--max-cells`** — A thread walks the $n_x \times n_y$ cells of its triangle's projected box (`t.span`), not only those inside. A long diagonal triangle has a huge box with few cells inside; with no cap, a single thread could walk millions of cells. With $2^{18} = 262\,144$, the cells beyond are counted (the effect's fourth counter, `counters[3]`) and the command warns on stderr; the surface left in those cells gets no Gaussians. With `--max-cells 1048576` ($2^{20}$) the cap is quadrupled.

**`kRunCeiling` and the 192 MB** — The ceiling is `2u << 20` $= 2^{21} = 2\,097\,152$ Gaussians per run. With records of six `float4` (96 bytes: the four base entries plus normal-metallic and roughness-transmission-UV), $2^{21} \times 96\,\text{B} = 192\,\text{MiB}$.

**Conversion in slices** — The prefix-sum pass knows each triangle's offset; the first one whose block does not fit under the ceiling is recorded with an atomic minimum (`InterlockedMin(counters[4], i)` in `m2sScan`) and writes nothing. The host relaunches the effect with `firstTriangle` at that index; `m2sCount` zeroes the earlier triangles. Since the output order is the mesh's (§3.4), concatenating the slices gives the same array as a run without a ceiling.

*References:* code: `plugins/mesh2splat/mesh2splat.slang` (`m2sCount`, `m2sScan`, field `firstTriangle`), `apps/athenea/src/CmdMesh2Splat.cpp` (`constexpr uint64_t kRunCeiling = 2u << 20`); `docs/decisions.md`, "A mesh past the ceiling is converted in slices, and a barb keeps its alpha".
:::

## 3.4 One Gaussian per cell

**The projection.** For a triangle $(\mathbf a, \mathbf b, \mathbf c)$ with geometric normal $\hat{\mathbf n}_g$,
the two axes $(i, j)$ along which $\hat{\mathbf n}_g$ points least define the projection; a point maps to
$\mathbf o(\mathbf x) = (x_i - b_{\min,i},\, x_j - b_{\min,j}) / R$, with $R$ the range in use (the model's, the
mesh's longest side, or the camera-derived box). A cell is $1/N$ of that unit square, i.e. $h = R/N$ in the world,
bounded as in Section 3.3. A Gaussian is emitted at every cell whose centre falls inside the projected triangle,
located by its barycentric coordinates $\boldsymbol\lambda$, at the world point
$\boldsymbol\mu = \lambda_0 \mathbf a + \lambda_1 \mathbf b + \lambda_2 \mathbf c$.

::: {.nota}
**Projection axes** — `projectionAxes` drops the axis of largest $|n_k|$ and keeps the other two, which are "the two axes along which the normal points least". $\mathbf o(\mathbf x)$ are the point's coordinates on those two axes, relative to the box's lower corner and divided by $R$, so the model fits in $[0,1]^2$.

**Cells and centres** — Cell $(c_x, c_y)$ has centre $((c_x + \tfrac12)/N,\, (c_y+\tfrac12)/N)$ in the unit square (`m2sCellCentre`).

**Barycentric coordinates** — With $\mathbf e_1 = \mathbf o_b - \mathbf o_a$, $\mathbf e_2 = \mathbf o_c - \mathbf o_a$ and $\mathbf r = \mathbf o - \mathbf o_a$,

$$
\lambda_1 = \frac{\mathbf r \times \mathbf e_2}{\mathbf e_1 \times \mathbf e_2}, \qquad
\lambda_2 = \frac{\mathbf e_1 \times \mathbf r}{\mathbf e_1 \times \mathbf e_2}, \qquad
\lambda_0 = 1 - \lambda_1 - \lambda_2,
$$

with $\times$ the 2D cross product (a scalar). The centre is inside if all three are $\ge 0$; the test is closed, so a centre on a shared edge counts in both triangles.

**Why $\boldsymbol\mu = \sum_k \lambda_k \mathbf v_k$ is exact** — The projection restricted to the triangle's plane is affine and bijective (unless the triangle is edge-on to its projection plane, which the dominant-axis choice prevents). Barycentric coordinates are invariant under affine maps, so those computed in 2D lift the point onto the 3D triangle without error.

*References:* code: `plugins/mesh2splat/mesh2splat.slang` (`projectionAxes`, `orthogonalUv`, `m2sCellCentre`, `m2sCellInside`, `m2sFlatAt`).
:::

**The sizes.** With $E = [\mathbf b - \mathbf a,\; \mathbf c - \mathbf a]$ and $O = [\mathbf o_b - \mathbf o_a,\;
\mathbf o_c - \mathbf o_a]$, the Jacobian of the projection's inverse is $J = E\,O^{-1} = [\mathbf j_u, \mathbf j_v]$
(guarded at $|\det O| > 10^{-24}$). The two in-plane standard deviations and the third are

$$
s_u = \min\!\left(\lVert \mathbf j_u \rVert\,\sigma\,\tfrac{1}{N},\; \ell_{\max}\right), \qquad
s_v = \min\!\left(\lVert \mathbf j_v \rVert\,\sigma\,\tfrac{1}{N},\; \ell_{\max}\right), \qquad
s_n = \varphi\,\min(s_u, s_v),
$$

where $\sigma$ is `--sigma`, $\ell_{\max}$ the triangle's longest edge and $\varphi$ `--flatness`. Three details follow
the original source: the Jacobian is over the triplanar coordinates, not over texture
coordinates; the $1/N$ factor is applied by the original's exporter, not by a shader (without it every Gaussian is
the size of the model); and the frame is not the Jacobian's columns but the triangle's longest edge
$\hat{\mathbf t}$, the interpolated shading normal $\hat{\mathbf n} = \operatorname{normalize}(\sum_k \lambda_k
\mathbf n_k)$, and their cross product. The covariance is therefore

$$
\Sigma = R_{\text{frame}}\,\operatorname{diag}(s_u^2, s_v^2, s_n^2)\,R_{\text{frame}}^{\!\top}, \qquad
R_{\text{frame}} = [\hat{\mathbf t}_\perp,\; \hat{\mathbf n}\times\hat{\mathbf t}_\perp,\; \hat{\mathbf n}],
$$

with $\hat{\mathbf t}_\perp$ the edge made orthogonal to $\hat{\mathbf n}$. Without the bound $\ell_{\max}$,
1 859 slivers of a BMW, seen almost edge-on by their own projection, produce Jacobian columns twenty orders of
magnitude too long and draw white spikes metres long.

::: {.nota}
**The Jacobian** — Over the triangle, the inverse of the projection is affine: $\mathbf x = \mathbf a + J(\mathbf o - \mathbf o_a)$. Requiring it to map $\mathbf o_b \mapsto \mathbf b$ and $\mathbf o_c \mapsto \mathbf c$ gives $J\,O = E$, that is $J = E\,O^{-1}$, with $E$ ($3\times2$) the edges in the world and $O$ ($2\times2$) the projected edges. Column $\mathbf j_u$ is the displacement in the world per unit of the projected coordinate $u$; a one-cell step is $1/N$ of that unit and measures $\lVert \mathbf j_u\rVert/N$ in the world. For a triangle parallel to its projection plane, $\lVert \mathbf j_u\rVert = R$ and $s_u = \sigma R/N = \sigma h$; tilted by an angle $\theta$ from that plane, the column along the tilt lengthens by $1/\cos\theta$ and the Gaussian stretches by as much as its samples spread apart.

**The guard $|\det O| > 10^{-24}$** — $O^{-1}$ carries $1/\det O$; a triangle almost edge-on in its projection has a tiny determinant and a Jacobian that blows up.

**The bound $\ell_{\max}$** — No Gaussian can be wider than its triangle's longest edge. It is a bound that always holds, independent of what the projection did; without it, 1 859 slivers of Blender's `bmw27` scene produced columns of absurd length and white spikes metres long crossing the frame.

**$s_n = \varphi \min(s_u, s_v)$** — The third axis, the thickness, is a fraction of the smaller width (see "Flatness and surfels").

**The local frame** — The Gaussian's axes are the longest edge made orthogonal to the shading normal by Gram–Schmidt, $\hat{\mathbf t}_\perp = \operatorname{normalize}(\hat{\mathbf t} - (\hat{\mathbf n}\cdot\hat{\mathbf t})\hat{\mathbf n})$, the vector $\hat{\mathbf n} \times \hat{\mathbf t}_\perp$ and the normal. $R_{\text{frame}}$ has those vectors as columns and is converted into a quaternion (`quaternionOfFrame`, Shepperd's method: the branch is chosen by the trace to avoid dividing by a small number). With $\Sigma = R\,\operatorname{diag}(s_u^2, s_v^2, s_n^2)\,R^\top$, the Gaussian is an elliptical disc lying in the tangent plane.

**What this frame approximates** — The widths $s_u, s_v$ come from the Jacobian's columns but are placed along the longest edge and its perpendicular, not along $\mathbf j_u, \mathbf j_v$. The exact footprint of an isotropic Gaussian of the projected grid, carried into the triangle's plane, would be $\Sigma_{\text{plane}} = (\sigma/N)^2\,J J^\top$, an ellipse aligned with $J$'s singular vectors. The two coincide for a triangle parallel to its projection plane and differ when the tilt is large; neighbouring Gaussians overlap enough that the difference does not open gaps in practice. It is the original code's criterion, kept.

*References:* [1], [12]; code: `plugins/mesh2splat/mesh2splat.slang` (`m2sTriangleOf`: `ju`, `jv`, `longestEdge`, `alongU`, `alongV`, `t.tangent`; `quaternionOfFrame`; `m2sWrite`, orthogonalisation of `alongU`); `docs/decisions.md`, "A gaussian cannot be bigger than the triangle it stands on" (within "Two assets out of Blender, and the two bugs they found").
:::

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

::: {.nota}
**$\sigma$, in cells** — Each Gaussian's in-plane standard deviation is $\sigma$ times the spacing between centres. On a grid of unit spacing, the sum of peak-1 Gaussians, $\sum_{i} e^{-\lVert\mathbf x - \mathbf c_i\rVert^2/2\sigma^2}$, is nearly constant for $\sigma \gtrsim 0.5$: by the Poisson summation formula it equals $2\pi\sigma^2$ with a relative ripple of order $e^{-2\pi^2\sigma^2}$, about $2\cdot10^{-4}$ at $\sigma = 0.65$ and $3\cdot10^{-9}$ at $\sigma = 1$. A converted surface is therefore a "blanket" of nearly uniform density.

**Why the ray tracer sees holes** — The "coverage" measurement is the fraction of the background the converted surface hides; "pixel" is the value of the quad lit head-on, roughly coverage × 0.5032 (the mesh's value). A Gaussian ray tracer (3DGRT [7]) evaluates each particle at the point of maximum response along the ray and composites the product transmittance; a ray passing between centres meets partial responses from several Gaussians and does not reach full opacity. The rasteriser, which projects ellipses and composites per pixel, is less affected (0.94 against 0.70 at $\sigma = 0.65$, according to `docs/decisions.md`): it is the two integrations that differ, not the cloud.

**The choice of 1.0** — At $\sigma = 1$ the coverage is 0.911 and the pixel 0.456 (9 % below the mesh); larger values cover more but blur a cell's details and add more overlap per point, which makes compositing more expensive. The original uses 0.65 because its renderer composites in its own way; here a surface that lets 30 % of the background through is not the conversion of a solid mesh.

*References:* [1], [7], [12]; code: `apps/athenea/src/CmdMesh2Splat.cpp` (option `--sigma`), `plugins/mesh2splat/mesh2splat.slang` (`sigmaX`, `sigmaY`); `docs/decisions.md`, "The width of a converted gaussian, and what it covers".
:::

**Flatness and surfels.** The original wrote $10^{-7}$ model units as the third size, which made a Gaussian's
thinness depend on the model's scale: a 66 mm pawn traced correctly while a 100-unit fox became a ghost under the ray
tracer. The third size is now the fraction $\varphi$ (default 0.1) of the smaller in-plane size. `--flatness 0`
writes surfels — Gaussians with one scale exactly zero, the 2D Gaussian-splatting disc — which the rasteriser draws
by a ray–plane intersection per pixel and the path tracer clamps to $10^{-4}$ of the widest axis.

::: {.nota}
**Flatness** — The original writes $10^{-7}$ units as the third size (its parameter is $\log(10^{-7})$, because 3DGS stores scales as logarithms). It is an absolute length, whereas the other two scales are in cells: the aspect ratio depends on the model's size. A 100-unit fox against a 0.066 pawn is about 1 500 times more extreme. For the rasteriser, which projects the covariance to 2D (EWA splatting [3]), the thickness hardly matters; for the ray tracer, which integrates density along the ray, a Gaussian that is too thin can become numerically invisible: the fox came out as a ghost. The fraction $\varphi = 0.1$ makes the aspect independent of scale.

**Surfels** — With `--flatness 0` the third scale is exactly zero and the covariance has rank 2: it is a Gaussian disc, the surfel of Pfister et al. [6] and the primitive of 2D Gaussian Splatting [5]. $\Sigma^{-1}$ does not exist, so it cannot be projected by EWA; the rasteriser draws it as 2DGS does, intersecting each pixel's ray with the disc's plane and evaluating the 2D Gaussian at the intersection. The path tracer needs a volume for its acceleration structure and clamps the null axis to $10^{-4}$ of the widest axis.

*References:* [3], [5], [6], [7]; code: `plugins/mesh2splat/mesh2splat.slang` (comment "THE THIRD SIZE IS A FRACTION OF THE OTHER TWO, not a length"); `docs/decisions.md`, "Simplify: fewer gaussians by pairs, and surfels".
:::

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

::: {.nota}
**Opacity versus coverage** — A surface's *coverage* $F$ is the fraction of what lies behind that the surface hides (a glass at 60 %, a mask at 50 %). A Gaussian's *opacity* $\alpha$ is its weight in alpha blending: at a point at normalised distance $d$ from its centre it hides $\alpha\,g$, with $g = e^{-d^2/2}$. Since several Gaussians lie over each point of the surface, the transmittance is a product, $T = \prod_i (1 - \alpha g_i)$, and writing $\alpha = F$ into each one covers much more than $F$.

**The derivation** — Taking logarithms and expanding $-\ln(1-x) = x + x^2/2 + O(x^3)$:

$$
-\ln T = \sum_i -\ln(1 - \alpha g_i) \approx \alpha \sum_i g_i + \frac{\alpha^2}{2}\sum_i g_i^2.
$$

On the grid of one-cell spacing, with $\sigma$ in cells, the sums are the integrals: $\sum_i g_i \approx \int e^{-r^2/2\sigma^2}\,\mathrm dA = 2\pi\sigma^2$ and $\sum_i g_i^2 \approx \int e^{-r^2/\sigma^2}\,\mathrm dA = \pi\sigma^2$. Imposing $T = 1 - F$:

$$
\frac{\pi\sigma^2}{2}\alpha^2 + 2\pi\sigma^2\alpha - L = 0, \qquad L = -\ln(1-F),
$$

whose positive root is the formula in the text, $\alpha = \bigl(\sqrt{4S^2 + 2SL} - 2S\bigr)/S$ with $S = \pi\sigma^2$.

**Values at $\sigma = 1$** — $F = 0.25 \to \alpha \approx 0.049$; $F = 0.5 \to \alpha \approx 0.112$; $F = 0.75 \to \alpha \approx 0.214$ (the $+1/255$ included). The other way round, writing $\alpha = 0.5$ into every Gaussian gives, to second order, $T \approx e^{-(2\pi\cdot0.5 + \pi\cdot0.25/2)} \approx 0.03$, a coverage of about 97 %, in line with the 96 % measured.

**The $1/255$ term** — 3DGS rasterisers discard a fragment when $\alpha g < 1/255$ (it would not change an 8-bit value) and so cut the Gaussian off at a finite radius; the mass beyond the cut is lost. Adding $1/255$ to $\alpha$ roughly pays back that loss, which is negligible for high opacities and dominant for very low ones (the thin glass of §3.8).

**Extremes** — $F \ge 0.999$ maps to 1, so an opaque surface stays exactly opaque; $F = 0$ gives 0.

**Reading the measurement** — With the first-order term alone, $T = e^{-2\pi\sigma^2\alpha}$ slightly overestimates coverage at high $F$ (0.770 instead of 0.75); the quadratic term corrects it to 0.752.

**`--opacity-cut`** — A cut-out mask (a feather's silhouette, a leaf) below 0.5 means "there is no surface here": the cell is neither counted nor written (`m2sCovered`), and the budget is not spent on the card's rectangle. If the material declares the mask as a threshold (UsdPreviewSurface's `opacityThreshold`, glTF's `alphaMode = MASK`), above the cut the coverage is 1 (`m2sMaskCoverage`).

*References:* [1]; code: `plugins/mesh2splat/mesh2splat.slang` (`m2sCoverageAlpha`, `m2sCovered`, `m2sMaskCoverage`; the header comment of `m2sCoverageAlpha` reproduces the derivation); `docs/decisions.md`, "A converted surface covers what its opacity says" and "Opacity is coverage, not glass"; the corresponding limitation in §3.12.
:::

**A triangle that catches no cell.** A triangle smaller than a cell need contain no cell centre; rasterised, it
vanishes. Without a fallback, that removes the Buick's grille and window rubbers (80 137 triangles become 34 011 Gaussians).
Such a triangle therefore gets one Gaussian at its centroid, sized by its own second moments (`m2sOwnSize`): the uniform
triangle's covariance $\Sigma_\triangle = \tfrac{1}{12}\sum_k \mathbf d_k \mathbf d_k^\top$ (corners about the
centroid, in the tangent plane), widened as a cell is, $\Sigma = 12\sigma^2 \Sigma_\triangle$, with the wide axis
clamped to $[0.05\,s_{\max}, s_{\max}]$ of the cell's size and the narrow axis to at least 0.02 of the wide. Without
it, a panel's rolled lip — a strip of sub-cell triangles — draws a band of cell-wide beads three pixels thick along
every Corvette edge where the path tracer draws a hairline. In the Corvette bakes, glass and tyres come out at about
one Gaussian per triangle at both 640 and 1280 px.

::: {.nota}
**The triangle that catches no cell** — With the cell-centre rule, a triangle smaller than a cell may contain no centre: a rasteriser skips it, which is correct for a rasteriser (another triangle represents the pixel), but here geometry would be lost. `m2sCount` assigns it one Gaussian at its centroid unless the mask says there is no surface there either.

**Second moments** — The covariance of the uniform distribution over a triangle, with the corners $\mathbf d_k$ measured from the centroid, is $\Sigma_\triangle = \tfrac1{12}\sum_k \mathbf d_k\mathbf d_k^\top$ (in general $E[\mathbf x\mathbf x^\top] = \tfrac1{12}(\sum_k \mathbf v_k\mathbf v_k^\top + \mathbf s\mathbf s^\top)$ with $\mathbf s = \sum_k\mathbf v_k$, and $\mathbf s = 0$ at the centroid).

**The $12\sigma^2$ widening** — A square cell of side $h$ has standard deviation $h/\sqrt{12}$ as a uniform distribution, and its Gaussian has $\sigma h$: the cell widens its square's deviation by $\sigma\sqrt{12}$ (2.25 at $\sigma = 0.65$, 3.46 at $\sigma = 1$). The triangle is widened in the same way, $\Sigma = 12\sigma^2\Sigma_\triangle$, so that a strip of small triangles covers like a strip of cells.

**The bounds** — The in-plane $2\times2$ covariance is diagonalised in closed form: eigenvalues $m \pm \sqrt{\tfrac14(a-c)^2 + b^2}$ with $m = (a+c)/2$, angle $\tfrac12\operatorname{atan2}(2b, a-c)$. The wide axis is clamped to $[0.05\,s_{\max}, s_{\max}]$ (never wider than a cell; never less than a twentieth, because a sliver with two nearly coincident corners gave Gaussians of $10^{-7}$ m that the merge tree fused into sizeless nodes) and the narrow axis to at least 0.02 of the wide.

**The band of beads** — A panel's rolled lip is a strip of triangles smaller than the cell; with one whole-cell Gaussian per triangle, the strip turned into a dark band three pixels wide along every edge. With the triangle's own size, the band is as wide as the lip.

**One Gaussian per triangle** — That the glass and tyres come out at one per triangle at both 640 and 1280 px indicates that their triangles are smaller than the cell in both cases: their Gaussian count is set by the mesh, not by the resolution.

*References:* code: `plugins/mesh2splat/mesh2splat.slang` (`m2sCount`, block "A TRIANGLE SMALLER THAN A CELL STILL EXISTS"; `m2sOwnSize`; `m2sCellEmit` with `caught = false`).
:::

**Deterministic order.** The original appended with an atomic counter, so two conversions of the pawn produced the
same 729 073 Gaussians in different orders. The emit is now three passes — count each triangle's covered cells, an
exclusive scan in one 256-thread workgroup, write each triangle's Gaussians at its offset — sharing one function
(`m2sTriangleOf`) so that counting and writing cannot disagree. Two conversions are now byte-identical, a budget
keeps the first Gaussians in mesh order, and Gaussian $i$ is the same Gaussian from one pose to the next. A record
that decodes to nothing is written as an empty slot rather than dropped, so array indices stay aligned.

::: {.nota}
**The trouble with the atomic counter** — With `InterlockedAdd` each thread reserves its slot when it arrives, and the arrival order depends on the GPU's scheduling: two conversions give the same Gaussians in different orders. That makes it impossible to speak of "Gaussian $i$" in a stable way.

**Count, scan, write** — (1) `m2sCount`: each triangle counts its covered cells $c_t$. (2) `m2sScan`: exclusive prefix sum, $o_t = \sum_{t' < t} c_{t'}$, the triangle's offset. (3) `m2sEmit`: each triangle writes its Gaussians at $o_t, o_t + 1, \dots$. The prefix sum is done in a single group of 256 threads in three phases: each thread sums a contiguous run of triangles, thread 0 walks the 256 partials serially, and each thread walks its run again writing its running sum. It is not a work-efficient scan (Blelloch), but it avoids walking tens of thousands of triangles serially in a single thread.

**Shared `m2sTriangleOf`** — Everything a triangle decides (plane, Jacobian, cell, box, cell cap) comes from a single function called by both the counting and the writing pass; if they disagreed by a single bit, a triangle would write more or fewer slots than were reserved for it.

**Consequences** — Two byte-identical conversions; a budget that keeps the first Gaussians in mesh order (and not those that won a race); and a stable identity for each Gaussian across poses, which is what the skinning and per-Gaussian transfer of §5 need. A record that produces nothing (mask at zero, degenerate frame) is written as an empty zero-opacity slot instead of being dropped, so that the parallel arrays (records, rays, joints) stay aligned by index.

*References:* code: `plugins/mesh2splat/mesh2splat.slang` (comment on `M2sTriangle`, `m2sCount`, `m2sScan` with `kScanThreads = 256`, `m2sEmit`); `docs/decisions.md`, "The conversion's order is the mesh's, and a slot with nothing in it is kept".
:::

## 3.5 Where a surface stops or folds

A Gaussian is a cell wide, and on the Corvette a cell is a pixel of the conversion camera. A part narrower than that
— the flange a hood folds down into its gap — still received cell-wide Gaussians in its own plane; they reached out
of the gap over the paint beside it, carrying the gap's baked darkness, and every panel edge became a thick black
wavy line where the path tracer draws a thin one.

::: {.nota}
**The defect** — The flange is the strip that a panel's edge folds inwards into the gap, a few millimetres deep. Its Gaussians, a cell wide in their own plane, stick out of the gap towards the neighbouring paint; since almost no light reaches the gap, the bake leaves them dark, and the result is a thick, wavy black line along every panel edge. The same excess reach carries paint over the edge into the gap.

*References:* code: `shaders/athenea/usd/mesh2splat_edges.slang` (header comment, "WHERE A SURFACE STOPS OR FOLDS"); `docs/operations.md` §2.8 (option `--edge-reach`).
:::

**Hard edges** (`mesh2splat_edges.slang`). Per mesh, over all its pieces, every edge and corner is keyed by where its
points stand and radix-sorted on the device; each run of equal edge keys is one edge. An edge is *hard* when it is
open (one triangle uses it: the part stops or another mesh begins), when the two triangles sharing it fold by more
than $40^\circ$ (`kCreaseDegrees`), or when more than two triangles share it (non-manifold). Because keys are
positions over the whole mesh, a boundary between two materials on one smooth surface is not an edge. Each hard edge
becomes a line with an inside, $(\hat{\mathbf m}, c)$ with inside $\hat{\mathbf m}\cdot\mathbf x \ge c$ — the plane
through the edge, orthogonal to its triangle, facing the third corner — known to its own triangle as a bit and told
to every triangle touching either end (up to five lines each), since a Gaussian near a corner can stand beside the
next triangle's hard edge.

::: {.nota}
**Keying by position** — Each corner is identified by the bits of its three coordinates mixed with the MurmurHash3 finaliser, not by its index: a mesh with UV seams (duplicated corners with the same position and different UVs) is "welded" by position and its seams do not count as open edges. An edge's key combines the keys of its two ends without regard to order. The keys are sorted with a radix sort on the device; in the sorted list each run of equal keys is one edge, and the run's length is the number of triangles that use it.

**Hard-edge criteria** — *Open*: one triangle uses it (the surface's border, or the start of another mesh). *Fold*: the two triangles form a dihedral angle of more than $40^\circ$, that is, their normals satisfy $\hat{\mathbf n}_1\cdot\hat{\mathbf n}_2 < \cos 40^\circ \approx 0.766$. *Non-manifold*: three or more triangles share it. Since the keys are taken over all of the mesh's pieces, the border between two materials on one smooth surface has two triangles and is not a hard edge.

**The line with an inside** — Each hard edge is represented as a half-space $\hat{\mathbf m}\cdot\mathbf x \ge c$: $\hat{\mathbf m}$ is the component of (third corner − edge origin) perpendicular to the edge, normalised, and $c = \hat{\mathbf m}\cdot\mathbf v_0$. It is the plane that contains the edge and is perpendicular to the triangle, facing the triangle's interior.

**Propagation to neighbours** — A Gaussian near a corner can stand beside the hard edge of a neighbouring triangle sharing that corner; that is why each line is told to every triangle touching either of its ends (binary search in the sorted list of corners), up to five per triangle (`kLines`). For each piece the "Edges clip" is formed, an image with the shape of the mesh image: in each triangle's first entry, how many lines it received and the bits of its own hard edges; in the next five, the lines.

*References:* code: `shaders/athenea/usd/mesh2splat_edges.slang` (`m2sEdgeKeys`, `m2sEdgeRuns`, `tellEnd`, `m2sEdgeWrite`, `m2sEdgeCopy`, `kLines = 5`), `apps/athenea/src/CmdMesh2Splat.cpp` (`markEdges`, `kCreaseDegrees = 40.0F`), `modules/gpu/include/athenea/gpu/algo/RadixSort.h`.
:::

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

::: {.nota}
**The quantities** — In the Gaussian's tangent basis $(\mathbf u, \mathbf v)$, $C$ is its in-plane $2\times2$ covariance and $\mathbf m$ the line's normal projected onto the plane and normalised. $r^2 = \mathbf m^\top C\mathbf m$ is the Gaussian's variance in the direction perpendicular to the line, and $d = \hat{\mathbf m}\cdot\boldsymbol\mu - c$ the centre's signed distance to the line (positive towards the inside).

**The allowed size** — $a = \max(d/\rho, 0.05\,r)$: the deviation perpendicular to the line cannot exceed $d/\rho$. With $\rho = 1.5$, the edge lies 1.5 standard deviations from the centre: the Gaussian's profile there is $e^{-1.5^2/2} \approx 0.32$ and the fraction of its mass beyond the line is $1 - \Phi(1.5) \approx 6.7\,\%$. The minimum $0.05\,r$ prevents a Gaussian whose centre lies on the line from degenerating into a zero-width needle.

**The rank-one update** — $C' = C - \kappa\,(C\mathbf m)(C\mathbf m)^\top$ with $\kappa = (1 - a^2/r^2)/r^2$. Check: $\mathbf m^\top C'\mathbf m = r^2 - \kappa r^4 = a^2$. For a direction $\mathbf w$ conjugate to $\mathbf m$ with respect to $C$ ($\mathbf m^\top C\mathbf w = 0$), $C'\mathbf w = C\mathbf w$: the Gaussian keeps its extent along the line, so a row of Gaussians beside an edge still covers the strip.

**Interpretation** — It is exactly a Kalman filter update, or the product of the Gaussian with a 1D Gaussian in the direction $\mathbf m$: $C' = C - C\mathbf m\,(\mathbf m^\top C\mathbf m + \nu)^{-1}\mathbf m^\top C$ with "measurement" variance $\nu = a^2 r^2/(r^2 - a^2)$; one checks that $1/(r^2 + \nu) = \kappa$. Hence $C'$ remains symmetric positive definite. Unlike a full product, the centre does not move: only the shape changes.

**Ignored lines** — If the line's normal hardly lies in the Gaussian's plane ($\lVert\mathbf m\rVert^2 \le 0.25$ before normalising, more than $60^\circ$ out of the plane), the line does not constrain it: this is the case of the flange seen from the paint it folds off. If the centre lies behind the line by more than $0.25\,r$, the line belongs to the other side.

**Canonical order** — Successive updates do not commute exactly; the lines are sorted by their offset $c$ and then by their direction (insertion in `m2sHoldEdges`) so that the result does not depend on the order in which they arrived. At the end, $C'$ is diagonalised into a tangent and two scales, and the third size is recomputed with $\varphi$.

*References:* code: `plugins/mesh2splat/mesh2splat.slang` (`m2sHoldEdges`, `kHeldLines = 8`, call at the start of `m2sWrite`); `docs/operations.md` §2.8 (`--edge-reach`, a number from 0 to 8, default 1.5).
:::

**Closed shells.** The same pass counts edges not shared by exactly two triangles, one counter read back per mesh. A
mesh is marked closed when

$$
n_{\text{open}} < 2\sqrt{1.5\,T},
$$

$T$ its triangles ($1.5T$ approximates its welded edges), and its Gaussians carry
`primvars:athenea:splat:closed = 1`, which allows a player or a cut to drop nodes facing away. A fixed share of edges
(2 %) is not sufficient: an open sheet's border grows as the square root of its area, so a fine enough tube
passes as closed — with that rule the rims' barrels (1 %) are culled and light shows between the spokes. Against the
square root, the doors (no open edge), logos (1.4) and calipers (1.5) stay closed and rims (4.0), tyres (3.1), hood
(3.1) and bumper (4.4) do not: 102 of the Corvette's 192 meshes. A car's door is two skins; drawn without culling,
the inner skin sorts in front of the outer in patches, a net of dark cracks at small screen sizes.

::: {.nota}
**Closed shell** — A mesh is a closed surface (a manifold without boundary, *watertight*) if every edge is shared by exactly two triangles. On a closed, oriented surface, back faces are always hidden by front faces, so they can be discarded without changing the image: this is back-face culling. The `primvars:athenea:splat:closed` mark allows a player or a cut (`--cut-cull-back`, §6) to discard nodes whose cone of normals faces entirely away.

**The criterion $n_{\text{open}} < 2\sqrt{1.5\,T}$** — $n_{\text{open}}$ counts the edges not shared by exactly two triangles (open or non-manifold); a welded triangulated mesh has about $E \approx 1.5\,T$ edges. A real mesh is almost never perfectly closed (small holes, seams), so a tolerance is needed; what matters is how it scales. An open sheet's border grows as the square root of its area: a square sheet of $n\times n$ quads has $2n^2$ triangles, about $3n^2$ edges and $4n$ border edges, a ratio $4n/\sqrt{3n^2} = 4/\sqrt3 \approx 2.31 > 2$, and stays open. A tube of $n$ segments around and $m$ along has $2n$ border edges and about $3nm$ edges, a ratio $\tfrac{2}{\sqrt3}\sqrt{n/m}$: a wide, short tube (a rim's barrel) gives a high ratio and stays open.

**Why not a fixed share** — With "fewer than 2 % open edges", the same tube passes as closed as soon as $2n/(3nm) = 2/(3m) < 0.02$, that is with more than about 33 rings, even though it is open at both ends. A rim's barrel (1 %) was marked closed, its inner face was culled and light showed between the spokes.

**The figures in parentheses** — They are each mesh's ratio $n_{\text{open}}/\sqrt{1.5\,T}$ against the threshold 2: doors 0, logos 1.4 and calipers 1.5 (closed); rims 4.0, tyres 3.1, hood 3.1 and bumper 4.4 (open).

**The door's two skins** — 3DGS rasterisation sorts Gaussians by the depth of their centre, not per pixel; two very close surfaces (a door's inner and outer skin) interleave badly in patches and leave dark cracks at small screen sizes. With the door marked closed, the skin facing away is culled.

*References:* code: `apps/athenea/src/CmdMesh2Splat.cpp` (`markEdges`, block "A COUNTER BACK": `closed_[h.piece] = double(open) < 2.0 * std::sqrt(meshEdges)`), `modules/usd/include/athenea/usd/Export.h` (`primvars:athenea:splat:closed`); `docs/operations.md` §2.4.1 (`--cut-cull-back`, `--mark-closed`).
:::

**Measured.** The Corvette's TX base at 640 px of `/World/Camera`, rebaked with edges held (default 1.5) and as a
control with `--edge-reach 0`, both 5 380 804 Gaussians; day frame at 1920 × 1080 against the mesh path trace:

| bake | relMSE (HDR) | PSNR (display) | SSIM |
|---|---|---|---|
| previous base (earlier build) | 0.0900 | 21.83 dB | 0.8555 |
| control, edges not held | 0.0902 | 21.86 dB | 0.8554 |
| edges held, $\rho = 1.5$ | 0.0887 | 21.93 dB | 0.8567 |

The global gain is small because the edges are a small fraction of the frame; the visible effect is on the panel
gaps.

::: {.nota}
**The metrics** — *relMSE* (relative mean squared error, in HDR): the mean of $(x - y)^2/(y^2 + \epsilon)$ over the pixels, with $y$ the reference; it weighs errors in dark and bright regions equally. *PSNR* (display): $10\log_{10}(1/\text{MSE})$ over the image already transformed for display, in dB; +1 dB amounts to reducing the MSE by 21 %. *SSIM* (Wang, Bovik, Sheikh, Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity", IEEE TIP 2004): local structural similarity of luminance, contrast and structure, between 0 and 1.

**Reading** — Between the control and the bake with edges held, relMSE drops by 1.7 % (0.0902 → 0.0887), PSNR rises by 0.07 dB and SSIM by 0.0013. Both bakes have exactly 5 380 804 Gaussians: the pass changes shapes, not counts. The "earlier build" row differs from the control by changes unrelated to this pass.

*References:* §2.15 of the paper (image comparison on the GPU); `docs/operations.md` §2.11 (`athenea compare`).
:::

## 3.6 Material capture

Each Gaussian record carries position, coverage-derived opacity, three sizes, a rotation and a colour, followed —
depending on what the cloud needs — by the shading normal and metallic, roughness, transmission and texture
coordinates; four joints and their gradients; three relief entries for displacement; an emission entry; four layer
entries; and the shape operator.

::: {.nota}
**A Gaussian's record** — In the effect's output image, each Gaussian takes several consecutive `float4` entries (`recordPixels`): (0) position and opacity; (1) the three sizes; (2) the rotation as a quaternion; (3) the colour; with the PBR channels, (4) shading normal and metallic and (5) roughness, transmission and UV; with skinning, (6–8) joints, weights and gradients; with displacement, three more (flat point, its normal, the relief's normal); with emission, one; with layers, four (specular, coat, sheen, coat darkening); and with curvature, one with the shape operator. `mesh2splat_gather` reorders these entries into the cloud's layout (the fourteen canonical floats, then metallic, roughness, transmission, normal and emission, the harmonics the bake will write and, at the end, the thirteen floats of the layers).

*References:* code: `plugins/mesh2splat/mesh2splat.slang` (`m2sWrite`, computation of `own`), `shaders/athenea/usd/mesh2splat_gather.slang` (header comment, `m2sGather`).
:::

**Base.** The colour is the material's base colour times its base weight, times the albedo map where there is one
(decoded to linear). Metallic and roughness are the material's constants, multiplied by the packed map's channels
where there is one (blue metallic, green roughness), as glTF multiplies its factors; a missing channel is a factor
of one. The original defaults to $(0.1, 0.5)$ from the map alone, so every material without a map becomes the same
plastic; with the material's own values passed, a converted quad under a head-on light reads 0.5009 against the
mesh's 0.5032.

::: {.nota}
**Base colour** — It is the product of the base weight (`base` or `base_weight`), the base colour and the albedo map's texel, decoded to linear by `texture_rows`. It is an albedo, not a radiance: the bake (§4) turns it into light.

**The metallic-roughness map** — glTF packs occlusion, roughness and metallic into the R, G and B channels of a single texture (the "ORM"); the glTF specification multiplies the material's factors by the map's channels, and the conversion does the same. A missing channel is a factor of 1.

**The original's defect** — It took $(0.1, 0.5)$ for metallic and roughness from the map and ignored the material's values: every material without a map came out as the same plastic. With the material's own values, the test quad gives 0.5009 against the mesh's 0.5032 (0.46 % below).

*References:* glTF 2.0 specification (Khronos, `pbrMetallicRoughness`); code: `plugins/mesh2splat/mesh2splat.slang` (`m2sWrite`: "Blue is metallic, green is roughness: their packing").
:::

**Shading normal.** A normal map is read in the frame its UVs give — $\partial\mathbf p/\partial u$ from the
triangle's positions and UVs, orthonormalised against the normal, as the mesh shader does — and the result is kept as
the Gaussian's shading normal (`primvars:athenea:splat:normal`) beside its geometric frame. Read against the longest
edge instead, every facet of the pawn's marble turns its veins by its own angle. The disc
itself is not turned unless `--normal-map-turns` is given, which would open the surface where the relief is steep.

::: {.nota}
**A normal map's frame** — A tangent-space normal map encodes the normal as $(2t - 1)$ in the basis $(\mathbf T, \mathbf B, \mathbf N)$, with $\mathbf T = \partial\mathbf p/\partial u$ the direction in which the texture's $u$ coordinate grows. With edges $\mathbf e_1, \mathbf e_2$ and UV differences $(\Delta u_1, \Delta v_1)$, $(\Delta u_2, \Delta v_2)$:

$$
\mathbf T = \frac{\mathbf e_1\,\Delta v_2 - \mathbf e_2\,\Delta v_1}{\Delta u_1\Delta v_2 - \Delta u_2\Delta v_1},
$$

orthonormalised against the interpolated normal. If the UVs are degenerate, the branchless orthonormal basis of Duff et al. ("Building an Orthonormal Basis, Revisited", JCGT 2017) is used. The bitangent is $\mathbf N \times \mathbf T$, the same frame the mesh shader builds, so the Gaussian and the mesh read the relief alike.

**The error it corrects** — Read against the triangle's longest edge (the Gaussian's own axis, unrelated to how the map was painted), each triangle turned the relief by its own angle: on the pawn's marble, each facet lit its veins from a different side.

**Shading normal versus geometric frame** — The perturbed normal is stored separately (`primvars:athenea:splat:normal`) and the disc stays lying in the triangle's plane. Turning the disc towards the map's normal (`--normal-map-turns`) would tilt neighbouring discs in different directions and open slits where the relief is steep.

**A detail to watch** — The bitangent is taken without the handedness sign that MikkTSpace tangents carry (`tangent.w`); on mirrored UVs the map's green channel would be read inverted. It is consistent with the mesh shader, which does the same.

*References:* code: `plugins/mesh2splat/mesh2splat.slang` (`m2sMapTangent`, `m2sWrite`), `modules/usd/src/Export.cpp` (`primvars:athenea:splat:normal`); `docs/decisions.md`, "A normal map is read in the frame its UVs give" and "A gaussian keeps its shading normal apart from its frame".
:::

**Layers.** A Gaussian carries the specular weight, colour and index; the coat's weight, roughness and
index; the sheen's colour times its weight and its roughness; and the coat darkening switch — OpenPBR's units, three
words per splat on the device. Up to three maps on these inputs are sampled per Gaussian (`m2sLayersAt`). This is
required because the Corvette's paint (`Car_Paint_Main`) is OpenPBR metalness 1 over a base of about 0.05 under a
clear coat of weight 1: everything the eye sees of it is the lacquer, and without the coat a converted car comes out
nearly black.

::: {.nota}
**OpenPBR's layered model** [30] — The surface is a stack: a base (dielectric, metallic, subsurface, transmissive), a dielectric specular layer over it (weight, colour, index), a clear coat (a lacquer layer with its own weight, roughness and index; `coat_darkening` controls how much the internal reflection under the coat darkens the base) and a sheen (fibre retro-scattering, with colour, weight and roughness [28]). Each layer attenuates what it lets through to the one below according to its Fresnel.

**Three words per splat** — On the device (`GpuSplats::lobes`), the thirteen layer values (specular weight, colour and index; coat weight, roughness, index and darkening; sheen colour and roughness) are quantised into three 32-bit words per Gaussian, a few bits per value and in the units of OpenPBR's own inputs. The index takes a byte in which 1.5 is an exact step, so a cloud without layers and one with the default layers (`plainLobes`) give the same image. Coat darkening is a switch: on from 0.5 upwards.

**`m2sLayersAt`** — Up to three maps can replace layer constants; each carries a target (1 specular weight, 2 specular colour, 3 coat weight, 4 coat roughness, 5 sheen colour, 6 sheen weight, 7 sheen roughness) and a channel.

**`Car_Paint_Main`** — Metalness 1 over an almost black base colour (about 0.05) under a coat of weight 1. A metal with $F_0 \approx 0.05$ reflects very little at normal incidence; what the eye perceives is the coat's reflection (a dielectric of index 1.5–1.6, $F_0 \approx 0.04$–0.05, but smooth and with Fresnel rising towards the edge). Without the coat, the paint is a dark, rough metal and the car comes out nearly black.

*References:* [28], [30]; code: `shaders/athenea/common/packing.slang` (`SplatLobes`, `plainLobes`), `plugins/mesh2splat/mesh2splat.slang` (`m2sLayersAt`, `m2sLayerTexel`, fields `specular`, `coat`, `sheen`, `coatDarkening` of `M2sParams`); `docs/decisions.md`, "A gaussian carries what its material layers over the base (proposal 026)" and "Maps on the layers".
:::

**Schlick metals.** OpenPBR and glTF metals are MaterialX `generalized_schlick_bsdf` (colour head-on, specular colour
at $82^\circ$), not a conductor of the artistic index; for a dark metal the two differ by a factor of two at grazing
angles (at $60^\circ$: 0.172 for the conductor, 0.077 for the Schlick from $F_0 = 0.047$). The material's kind is
written per Gaussian (`primvars:athenea:splat:schlickMetal`) and the renderer reflects it with the DFG fit of its two
ends. Treating every OpenPBR metal as dielectric polish in the bake has a measured cost: a Mustang's front bumper
stores the constant that decodes to black in all 27 016 Gaussians; with Schlick lobes of $F_0 > 0.2$ kept as body,
the bumper's mean colour is (0.45, 0.51, 0.61).

::: {.nota}
**Two metal models** — `standard_surface` models a metal as a conductor whose complex index $(n, k)$ is derived from the reflectance colour and the edge colour (Gulbrandsen's "artistic IOR", "Artist Friendly Metallic Fresnel", JCGT 2014), with the exact conductor Fresnel. OpenPBR and glTF instead use MaterialX's `generalized_schlick_bsdf`: a Schlick interpolation between the head-on colour $F_0$ and white at $90^\circ$, with a correction so that at $82^\circ$ it equals the specular colour (Hoffman's F82-tint model, "Fresnel Equations Considered Harmful", 2019, adopted by Adobe Standard Material and OpenPBR):

$$
F_{\text{Schlick}}(\theta) = F_0 + (1 - F_0)(1 - \cos\theta)^5 \;\;(\text{without the F82 correction}).
$$

**The numbers** — At $60^\circ$, $(1-\cos 60^\circ)^5 = 0.5^5 = 0.03125$ and $F = 0.047 + 0.953\cdot0.03125 \approx 0.077$. The conductor with $n = 0.91$, $k = 0.41$ (the one that reproduces $F_0 \approx 0.046$) gives 0.172: twice as much. The difference grows towards the edge (at $70^\circ$, 0.312 against 0.165); since the paint is seen mostly on the car's flanks at grazing angles, a conductor brightened it by 40 %.

**How the renderer uses it** — Each Gaussian carries a `schlickMetal` bit and the renderer reflects its metal with a DFG fit of its own two ends. *DFG* is the precomputed split-sum term of environment lighting (Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013; Lagarde [19]): $\int f\cos\theta\,\mathrm d\omega \approx F_0\,A(r, \cos\theta) + F_{90}\,B(r,\cos\theta)$, with $A$ and $B$ tables or analytic fits in roughness and angle.

**The Mustang's bumper** — If the bake treats every OpenPBR metal as a dielectric lacquer, the metal's body receives no light and its Gaussians store a constant that decodes to black. Keeping Schlick lobes with $F_0 > 0.2$ (bright metals) as "body", the bumper recovers its mean colour (0.45, 0.51, 0.61).

*References:* [19], [24], [30], [31]; code: `shaders/athenea/material/mx/athenea_generalized_schlick_bsdf.slang`, `shaders/athenea/splat/splat_relight.slang` (`schlickMetal`, `environmentBrdfOf`, `ggxEnvSchlickMetal`), `modules/usd/include/athenea/usd/Export.h` (`primvars:athenea:splat:schlickMetal`); `docs/decisions.md`, "A car's paint is a Schlick metal, not a conductor".
:::

**Emission and transmission.** Emission is the material's colour times its map, linear and unbounded, in an entry of
its own. Transmission $T$ and transmission colour $\mathbf c_T$ are a channel the original did not have (glass comes
out opaque white). The colour is taken towards the transmission colour,

$$
\mathbf c \leftarrow \operatorname{lerp}\!\bigl(\mathbf c,\; \operatorname{lerp}(\mathbf 1, \mathbf c, w_T)\,\mathbf c_T,\; T\bigr),
$$

with $w_T = 1$ for glTF (transmission tinted by the base colour) and 0 for OpenPBR and standard_surface. The opacity
is the coverage mapping of Section 3.4 applied to $o\cdot o_{\text{mat}}\cdot\text{mask}\cdot
\operatorname{lerp}(1, g, T)$, $g$ being what a fully transmitting surface still covers (Section 3.8).

::: {.nota}
**Emission** — Linear, unbounded radiance (it may exceed 1), in an entry of the record of its own (`m2sEmissionAt`): the emission colour times the map, or one channel of the map as grey.

**Transmission** — $T \in [0,1]$ is the fraction of light the surface lets through instead of reflecting or diffusing it, and $\mathbf c_T$ the colour that tints that light. The original has no such channel: a glass material comes out as an opaque white Gaussian.

**The colour formula** — With $T = 1$ the colour becomes $\operatorname{lerp}(\mathbf 1, \mathbf c, w_T)\,\mathbf c_T$: for glTF ($w_T = 1$) the product $\mathbf c\,\mathbf c_T$, because in `KHR_materials_transmission` the transmission is tinted by the base colour; for OpenPBR and standard_surface ($w_T = 0$) only $\mathbf c_T$, because their base colour belongs to the opaque base and not to the transmission.

**The opacity** — The product $o\cdot o_{\text{mat}}\cdot\text{mask}\cdot\operatorname{lerp}(1, g, T)$ is a coverage (`--opacity`, the material's opacity, the mask and what a glass keeps), which the mapping of §3.4 turns into $\alpha$. For a solid glass with $T = 1$ and $g = 0.6$, the surface hides 60 % of the background.

*References:* code: `plugins/mesh2splat/mesh2splat.slang` (`m2sEmissionAt`, `m2sWrite`: `keptOpacity` and the interpolation with `transmissionColour.w`); `docs/decisions.md`, "A converted gaussian gives off what its material gave off" and "A glass gaussian's colour is its transmission's own".
:::

**The material roughness mark.** The stored roughness may be widened by the specular filter (Section 3.7). The
material's own roughness before widening is written separately, `primvars:athenea:splat:materialRoughness` (−1 under
a roughness map, read as sharp). The player uses it to decide whether the polish lobe answers its occluded part from
the baked reflected field (material roughness at least 0.3: paint, 0.34) or keeps it dark (chrome 0, rough metal
0.22); in the decimated tree it is packed into the `pbr` word (bit 26 present, bits 27–31 the value in five bits).

::: {.nota}
**`materialRoughness`** — The roughness stored in the record may have been widened by the specular filter (§3.7); the player also needs the material's, because it uses it to decide what to do with the occluded part of the polish lobe: if the material is rough enough (roughness $\ge$ 0.3, `kPolishFieldRoughness`), the lobe answers with the bake's *reflected field* (what the closed directions see, §4); if it is smoother (chrome 0, rough metal 0.22), it keeps it dark. Under a roughness map −1 is written, which is read as sharp.

**The packing** — In the decimated tree, the `pbr` word reserves bit 26 for "present" and bits 27–31 for the value in five bits (32 levels).

*References:* code: `modules/usd/include/athenea/usd/Export.h` (`materialRoughness`), `shaders/athenea/splat/splat_relight.slang` (`kPolishFieldRoughness = 0.3`), `shaders/athenea/splat/splat_project.slang` (use of `surface.materialRoughness`); `docs/decisions.md`, "The polish sees its ground where its material is rough enough".
:::

## 3.7 The specular filter (Toksvig widening)

A Gaussian has one normal. Where a strip of chrome curves under it, the path-traced pixel averages a highlight that
moves across the footprint and shows a thin line; the Gaussian shows the light its single normal meets, at full
brightness, over its whole footprint. On the Corvette, without filtering, the grille trim — four pixels high — comes out as a row of
white beads, and the chrome reads 1.6 times the path trace while a chrome ball validates within 1 %.

::: {.nota}
**The problem: one normal per Gaussian** — A path-traced pixel integrates the BRDF over all the normals that fall within its footprint; on a curved surface, the highlight moves across the pixel and the result is a thin line of medium brightness. A Gaussian evaluates a single normal, its centre's, and shows the highlight at full brightness over its whole footprint: a chrome strip four pixels wide comes out as a row of white beads. It is the same phenomenon as *specular aliasing* when normal maps are downsampled: the mean of the normals does not give the mean of the shading.

**The classical solution** — Replace the single normal by a wider distribution of normals, that is, increase the roughness according to the spread of normals lost by averaging: Toksvig [20] (estimates the variance from the length of the mean normal), LEAN mapping (Olano and Baker, I3D 2010), the frequency-domain filtering of Han et al. (SIGGRAPH 2007), and the geometric specular antialiasing of Kaplanyan et al. ("Filtering Distributions of Normals for Shading Antialiasing", HPG 2016) and Tokuyoshi and Kaplanyan (I3D 2019). A wider lobe roughly conserves the highlight's energy and spreads it over the footprint, as the path trace does.

**The chrome ball** — It validates within 1 % because its curvature is small relative to the cell: under each Gaussian the normal hardly turns. The error belongs to the thin strip, not to the material.

*References:* [20]; `docs/decisions.md`, "A reflection off a curved strip is the mean over the gaussian (Toksvig)".
:::

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

::: {.nota}
**The spread $s$** — $\angle(\mathbf n_{e_0}, \mathbf n_{e_1})/\lVert\mathbf p_{e_1} - \mathbf p_{e_0}\rVert$ is the normal's turn rate (radians per unit length, a curvature) along an edge; the largest of the three is taken. Multiplied by the Gaussian's wider size, it gives how many radians the normal turns across one standard deviation: an angular deviation.

**Why $\alpha'^2 = \alpha^2 + 2s^2$** — In GGX with Burley's parametrisation [27], $\alpha = r^2$ (perceptual roughness squared). For Beckmann-type microfacet distributions, the slope variance per axis is $\alpha^2/2$ (GGX has infinite variance and the same relation is used as an approximation). A small angular spread $s$ amounts to a slope $\tan s \approx s$, with variance $s^2$ per axis. Independent variances add: $\alpha'^2/2 = \alpha^2/2 + s^2$. In terms of perceptual roughness, $r' = (r^4 + 2s^2)^{1/4}$, clamped to $[0,1]$.

**Examples** — A smooth chrome ($r = 0$) with $s = 0.05$ rad (about $3^\circ$) goes to $r' = (0.005)^{1/4} \approx 0.27$; with $s = 0.1$ rad, to $0.38$. A flat triangle, or one with its three normals equal, gives $s = 0$ and is unchanged.

**`--specular-filter`** — The factor $k$ is 1 with `--transfer` and 0 otherwise, so that relight and baked-radiance conversions stay as they were.

**The coat** — It is widened in the same way, unless the cloud carries the shape operator, which handles the turn differently (next paragraph).

*References:* [20], [24], [25], [27]; code: `plugins/mesh2splat/mesh2splat.slang` (`m2sSpread`, `m2sWidened`, and in `m2sWrite` the condition `spread > 0.0 && params.curvatureEntry == 0` for the coat), `shaders/athenea/usd/mesh2splat_bake.slang` (the same formula in the bake).
:::

**The shape operator.** For a transfer the conversion also writes, in the Gaussian's own axes $(\mathbf u, \mathbf
v)$, the symmetric shape operator $S$ with $\mathrm d\mathbf n = S\,\mathrm d\mathbf x$, solved from the corners'
normal differences against their position differences (`m2sShape`, stored as $(S_{uu}, S_{uv}, S_{vv})$). The
renderer uses it to turn the reflection across the footprint pixel by pixel (the per-pixel "sharp lobe"). Widening
the coat as well would blur twice, so where $S$ is written the coat is not widened, and the player reads the sharp
lobes at the unwidened roughness $r_0 = \max(r'^4 - 2s^2, 0)^{1/4}$ (`unwidened`), with $s$ carried per Gaussian.

::: {.nota}
**The shape operator** — Also called the Weingarten map: the linear map of the tangent plane into itself that gives how the normal changes when moving over the surface, $\mathrm d\mathbf n = S\,\mathrm d\mathbf x$. It is symmetric; its eigenvalues are the principal curvatures $\kappa_1, \kappa_2$ and its eigenvectors the principal directions; $\operatorname{tr}S/2$ is the mean curvature and $\det S$ the Gaussian curvature.

**How it is solved (`m2sShape`)** — With $\mathbf d_1 = \mathbf n_1 - \mathbf n_0$, $\mathbf d_2 = \mathbf n_2 - \mathbf n_0$ (unit normals) and the edges $\mathbf e_1, \mathbf e_2$, all projected onto the Gaussian's axes $(\mathbf u, \mathbf v)$, one requires $S\,[\mathbf e_1\;\mathbf e_2] = [\mathbf d_1\;\mathbf d_2]$, that is $S = D\,A^{-1}$ with $A$ and $D$ $2\times2$ matrices. The solution is not symmetric in general (linear interpolation of normals is not an exact surface) and is symmetrised by averaging the cross terms: $(S_{uu}, \tfrac12(S_{uv}+S_{vu}), S_{vv})$ is stored. It is constant per triangle.

**The per-pixel "sharp lobe"** — With $S$, the renderer evaluates at each pixel of the footprint the normal $\mathbf n(\Delta\mathbf x) \approx \mathbf n + S\,\Delta\mathbf x$, where $\Delta\mathbf x$ is the pixel's offset from the centre in the Gaussian's plane. The reflection turns across the Gaussian as on a curved mirror, and the result resembles the path trace's thin line rather than a bead.

**Why not widen as well** — Widening the coat and also turning the reflection counts the same curvature twice: the hood's reflections came out soft where the path tracer draws them sharp. Where $S$ is written, the coat keeps its roughness, and the player recovers the sharp lobes' unwidened roughness by inverting the formula: $r_0 = \max(r'^4 - 2s^2, 0)^{1/4}$ (function `unwidened`), with $s$ stored per Gaussian.

*References:* do Carmo, *Differential Geometry of Curves and Surfaces* (1976), ch. 3 (the Gauss map and the shape operator); code: `plugins/mesh2splat/mesh2splat.slang` (`m2sShape`, `curvatureEntry`), `shaders/athenea/splat/splat_relight.slang` (`unwidened`), `shaders/athenea/splat/splat_project.slang` (use of `unwidened(coat, slope.bakeSpread)`); `docs/decisions.md`, "A TX transfer's reflection turns across the gaussian (proposals 044, 046, 049)" and "A sharp coat is read per pixel".
:::

**Measured.** At 640 px the coat came out at roughness 0.17–0.24 with 5–14 % high-pass on folds; at 1280 px the
spread under each Gaussian is smaller and the coat's roughness comes out at exactly 0, so the widening matters less as
the bake gets finer. Turning it off is nevertheless worse (Section 3.11): with `--specular-filter 0` a dashed dark
line appears on the sill's fold and the chrome strip sparkles.

::: {.nota}
**Reading the figures** — For a smooth coat ($r = 0$), $r' = (2s^2)^{1/4}$ and $s = r'^2/\sqrt2$: the roughness of 0.17–0.24 at 640 px corresponds to a spread of 0.020–0.041 rad (1.2°–2.3°) under each Gaussian. Doubling the resolution halves the Gaussian's size and also $s$; by the formula, $r'$ would drop only by a factor $1/\sqrt2$ (to about 0.12–0.17), not to zero.

**A remark on "exactly 0"** — In the code, the coat is not widened when the cloud carries the shape operator (`spread > 0.0 && params.curvatureEntry == 0` in `m2sWrite`). That the coat's roughness comes out at exactly 0 at 1280 px is consistent with that bake having written the curvature: at that resolution the bake no longer widens the coat. The smaller spread explains why the widening matters less; the exact zero is explained by the shape-operator rule.

**"High-pass"** — Refers to the high-frequency energy (the sparkle from one fold to the next) measured on the folds; the 5–14 % figure is not detailed further in the text.

*References:* §3.11 of the paper (variants A/B/C); code: `plugins/mesh2splat/mesh2splat.slang` (`m2sWidened`, `m2sWrite`).
:::

## 3.8 Glass

A Gaussian cannot refract, but it can let what is behind it through by its opacity, reflect through its specular
lobe, and carry an index of refraction that the renderer bends by (`primvars:athenea:splat:ior`, one per cloud; with
layers each Gaussian bends by its own specular index). The conversion's job is to decide, per transmitting piece,
whether the glass is a *solid* (a body: a ball, a lens) or a *thin wall* (a sheet or slab whose two faces bend
nothing), because the two are drawn differently.

::: {.nota}
**What a Gaussian can do with glass** — Alpha blending is straight-line transmission: what lies behind is seen attenuated but not displaced. Refraction requires bending the ray, which the rasteriser does not do; the renderer approximates it by reading the background in a direction bent according to the index (`primvars:athenea:splat:ior`, one per cloud; with layers, each Gaussian's own specular index).

**Solid and thin wall** — A *solid* (a ball, a lens) refracts on entering and leaving through non-parallel faces: it displaces the image of the background, inverts it or concentrates it. A *thin wall* (OpenPBR `geometry_thin_walled`, standard_surface `thin_walled`) models a sheet whose two faces are parallel and very close: the ray leaves parallel to the incoming one with a negligible lateral offset, so there is no net bending and what remains are the reflections off the two faces. The two classes are drawn with different models: the solid with the transfer's lens model (§4), the thin wall as an almost transparent layer that only reflects.

*References:* [24], [30]; code: `modules/usd/include/athenea/usd/Export.h` (`primvars:athenea:splat:ior`); `docs/decisions.md`, "Translucency: what a gaussian can do about glass" and "A converted glass bends, and its cloud says which schema it carries".
:::

**Classification** (`classifySheets`, `mesh2splat_sheet.slang`), for every transmitting piece whose material does
not already declare itself thin-walled. On the device, the piece's edges are keyed by position, sorted, and their
runs counted; the signed volume $V$ (tetrahedra to the origin) and the area $A$ are summed. Then

- *open*: more than $1/200$ of the edges have one triangle, or any edge has more than two;
- *slab*: closed, and the thickness $2V/A$ is below four of the model's cells or below $0.02\sqrt A$
  ($2V/A$ is a slab's gap and two thirds of a ball's radius).

An open or slab piece is read thin-walled — unless it is *tinted* (transmission colour luminance below 0.9), which
stays solid for now, since a thin wall lets the background through grey. The Corvette's glass consists of
closed slabs 3–4 mm thick, which the edge rule alone calls solid; the thickness rule reads
them. Each decision is printed, e.g. for the rear light lens: "tinted, kept solid glass: 200 of its 17284 edges open,
0 shared by more than two; 1.199 thick (2V/A) against a cell of 0.007921". `--thin-glass MAT` and `--solid-glass MAT`
override per material.

::: {.nota}
**Edges** — As in §3.5, edges are keyed by the position of their points (MurmurHash3 finaliser), sorted, and their runs counted: thus a sphere with a UV seam that repeats points is still closed. *Open* if more than $1/200$ (0.5 %) of the edges have one triangle (a tolerance so that a closed glass with an accidental hole stays solid) or if any has more than two.

**Volume and area** — By the divergence theorem, the volume enclosed by a closed oriented surface is the sum of the signed volumes of the tetrahedra each triangle forms with the origin, $V = \tfrac16\sum \mathbf a\cdot(\mathbf b\times\mathbf c)$, and the area $A = \tfrac12\sum\lVert(\mathbf b - \mathbf a)\times(\mathbf c - \mathbf a)\rVert$. Both sums are reduced in one device workgroup.

**The thickness $2V/A$** — For a slab with faces of area $A_c$ a distance $t$ apart (neglecting the rim), $V = A_c t$ and $A \approx 2A_c$, so $2V/A = t$. For a ball of radius $r$, $2V/A = 2\cdot\tfrac43\pi r^3/(4\pi r^2) = \tfrac23 r$. The same measure gives a slab's thickness and a fraction of a thick body's radius. *Slab* if $2V/A$ is less than four of the model's cells or than $0.02\sqrt A$ (one fiftieth of the glass's own size).

**Tinted** — Luminance of the transmission colour with the Rec. 709 coefficients, $0.2126R + 0.7152G + 0.0722B < 0.9$. A thin wall lets the background through grey (with its coverage, untinted), so a tinted glass is kept solid, carrying the tint in the transfer's field. The Corvette's tinted panes have a transmission colour of 0.52.

**The example message** — 200 of 17 284 open edges exceeds $1/200$ ($200\cdot200 = 40\,000 > 17\,284$): the lens is classed as open, but it is tinted and is kept solid. Its thickness, 1.199, is far above the cell of 0.0079: by the thickness rule it would be solid anyway.

**The Corvette's glass** — Closed slabs 3–4 mm thick with all their edges welded: the edge rule calls them solid; the thickness rule, slabs.

*References:* code: `apps/athenea/src/CmdMesh2Splat.cpp` (`classifySheets`, `thinGlassOf`, options `--thin-glass`, `--solid-glass`), `shaders/athenea/usd/mesh2splat_sheet.slang` (`m2sSheetEdges`, `m2sSheetRuns`, `m2sSheetSum`); `docs/decisions.md`, "A windscreen modelled as one surface is a sheet (`--thin-glass`)".
:::

**Coverage.** A solid glass covers `--glass-opacity` of what is behind it (by default 1 on a stage lit by domes
alone and 0.6 where a lamp or a sun stands: a TX lens answers its background from the domes alone, so under a lamp a
glass that covered fully hid the light shining through it). A thin
wall covers exactly what the sheet reflects head-on, both faces with inter-reflection:

$$
g_{\text{thin}} = \min\!\left(\frac{2R}{1+R},\; 0.998\right), \qquad R = \left(\frac{n-1}{n+1}\right)^2,
$$

and the frame scales its reflection back by $1/R$. At the windscreen's index of 1.16 that is an opacity of 0.0055,
and the rasteriser draws such sheets at an opacity of at least 0.1 with colour scaled down by the same factor, which
keeps the product $\alpha\,\mathbf c$ — the reflection — intact (windshield relMSE 2.26 → 0.89 under the workshop
sky).

::: {.nota}
**The solid glass's coverage** — In the TX bake, a lens's image (the sky seen through its two faces) is computed from the environment domes alone. Under a lamp or a sun, a glass that hid its whole background would hide the direct light passing through it (the lit ground behind a glass ball); with 0.6, 40 % of the real background passes through unbent. On a stage lit by domes alone, the lens's image is complete and 1 is used.

**The thin wall: $2R/(1+R)$** — With Fresnel reflectance at normal incidence $R = ((n-1)/(n+1))^2$ at each face and incoherent multiple reflections between them, the sheet's total reflectance is the series

$$
R + (1-R)^2 R + (1-R)^2 R^3 + \dots = R + \frac{(1-R)^2 R}{1 - R^2} = \frac{2R}{1+R}.
$$

What the sheet does not reflect it transmits in a straight line, exactly what a Gaussian's alpha blending does over its background; that is why the thin wall's coverage is what it reflects, and the frame scales the colour by $1/R$ (`thinWallGain`) so that the Gaussian shows that reflection. For $n = 1.5$, $R = 0.04$ and $g = 0.0769$. The 0.998 cap avoids full coverage.

**The windscreen's figures** — With $n = 1.16$, $R = (0.16/2.16)^2 \approx 0.0055$ and $1/R \approx 182$; the sheet's coverage is $2R/(1+R) \approx 0.0109$, and the coverage mapping of §3.4 at $\sigma = 1$ gives $\alpha \approx L/(2S) + 1/255 \approx 0.0017 + 0.0039 \approx 0.0057$ per Gaussian. The 0.0055 in the text coincides with $R$ and, up to rounding, with that $\alpha$; `docs/decisions.md` describes it as "what it reflects head on, 0.0055".

**`kSheetAlpha`** — With opacities this low, the $1/255$ discard and the radius cut removed most Gaussians before shading them, and the survivors were scattered dots with 182 times their colour. The rasteriser draws these sheets with $\alpha' = \max(\alpha, 0.1)$ and colour $\mathbf c' = \mathbf c\,\alpha/\alpha'$: the product $\alpha\mathbf c$, which in premultiplied blending is what the Gaussian adds (the reflection), does not change. The cost is that each Gaussian hides 10 % of what lies behind, which darkens the interior seen through the windscreen. The windscreen's relMSE goes from 2.26 to 0.89 under the workshop sky.

*References:* code: `plugins/mesh2splat/mesh2splat.slang` (`m2sGlassCovers`), `apps/athenea/src/CmdMesh2Splat.cpp` (`glassOpacity`: `stageHasLamps() ? 0.6 : 1.0`), `shaders/athenea/splat/splat_project.slang` (`kSheetAlpha = 0.1`), `shaders/athenea/splat/splat_relight.slang` (`thinWallGain`); `docs/operations.md` §2.8 (`--glass-opacity`); `docs/decisions.md`, "A solid glass covers 0.6 again" and "A thin sheet of glass is drawn as a layer".
:::

## 3.9 Shadow and light catchers

The ground is not converted as a surface; its interaction with the object is converted instead, so the playback is
Gaussians only.

::: {.nota}
**Shadow catcher** — A VFX compositing term: a surface that does not draw itself and only collects the shadow (and sometimes the reflections) of a CG object in order to integrate it over a background plate. Here the ground is converted not as a surface but as its interaction with the object, so the playback is Gaussians only, with no ground geometry in the player.

*References:* `docs/decisions.md`, "The ground's shadow as gaussians: a shadow catcher (task PLAY-G)".
:::

**Shadow catcher** (`--shadow-catcher --prim OBJECT`). The ground is found automatically (the largest flat mesh
outside the object whose top is at the object's bottom; `--catcher-ground` names one). A patch of quads — the
object's footprint widened by `--catcher-margin` heights (1.5), split into a grid so that no triangle exceeds
`--max-cells` — is added to an in-memory copy of the stage and converted alone at a cell of a hundredth of the object's
height, as a TX transfer whose rays meet the object and the ground. At draw time the gaussian's opacity is
$-\ln(\text{ratio})/2\pi$, where the ratio is light reaching it with the object against without, since a lattice of
$\sigma = 1$-cell Gaussians sums to $2\pi$ and a layer of opacity $a$ leaves $e^{-2\pi a}$. The Corvette's: a
5.9 × 8.3 m patch, 11 × 11 quads, 309 k Gaussians at a 1.26 cm cell, baked in 79 s. Against the path trace:

| | under the car (GT 0.182) | bumper contact (GT 0.043) | open ground (GT 0.217) |
|---|---|---|---|
| mesh ground's own shadow map | 0.182 | 0.041 | 0.215 |
| catcher, depth 2.4, two triangles | 0.144 | 0.002 | 0.206 |
| catcher, depth $2\pi$, grid | 0.192 | 0.030 | 0.220 |

::: {.nota}
**The ground and the patch** — The ground is located as the largest mesh that is flat along the vertical axis, outside the object, whose top face is at the object's bottom and which extends beneath it. The patch covers the object's footprint widened by `--catcher-margin` heights on each side. With the Corvette (about 1.26 m high, 2.0 m wide and 4.5 m long), $1.5 \times 1.26 \approx 1.9$ m per side gives roughly $5.9 \times 8.3$ m.

**Why a grid of quads** — The whole patch in two triangles would give each a projected box of about $470 \times 660$ cells of 1.26 cm (more than 300 000), above `--max-cells`; with two triangles a third of the patch was left unsampled. With $11\times11$ quads, each triangle walks a few thousand cells. The total, $5.9\times8.3\,\text{m}^2/(0.0126\,\text{m})^2 \approx 308\,000$, matches the 309 k Gaussians.

**The opacity $-\ln(\text{ratio})/2\pi$** — Each catcher Gaussian is a TX transfer: it stores how much light reaches it with the object present, for any sky. At draw time, the ratio between the light with the object and the light without it (the sky's irradiance at its normal plus the sun) is what remains; the Gaussian is drawn black with an opacity such that a layer of them lets exactly that ratio through. By §3.4, a lattice of $\sigma = 1$-cell Gaussians sums to $2\pi$ at every point and a layer of opacity $a$ lets $e^{-2\pi a}$ through (to first order); equating, $a = -\ln(\text{ratio})/2\pi$.

**The table** — The first estimate, a depth of 2.4 (the coverage a converted surface reaches) over two triangles, darkened 2–3 times more than the path trace and lost the bumper's contact (0.002 against 0.043). With $2\pi$ and the grid, all three zones come close to the reference and to the mesh ground's own shadow map.

**What it does not do** — A black layer can only remove light: the light the object bounces onto the ground, beyond what it takes away, is not drawn. That part is covered by the light catcher.

*References:* code: `modules/usd/src/ShadowCatcher.cpp` (`writeShadowCatcherStage`), `modules/usd/include/athenea/usd/ShadowCatcher.h`, `shaders/athenea/splat/splat_project.slang` (`splatProjectCatcher`, `kCatcherMark`), `shaders/athenea/splat/frame.slang`; `docs/operations.md` §2.8 (`--shadow-catcher`, `--catcher-ground`, `--catcher-margin`); `docs/decisions.md`, "The ground's shadow as gaussians: a shadow catcher (task PLAY-G)" and its subsection "Measured (M5 Pro, 1920 x 1080, the TX car and its catcher)".
:::

**Light catcher** (`--light-catcher`). The same patch, reaching `--light-reach` object lengths (3) ahead and behind,
at a cell of a thirtieth of the height, converted as a radiance bake per light group with the dome black; drawn
additively, it carries the low beam's pool and the tail lamps' glow.

::: {.nota}
**Light catcher** — The additive complement of the shadow catcher: the same kind of patch, longer (`--light-reach` 3 object lengths ahead and behind, because headlamp light reaches far) and coarser (a cell of 1/30 of the height, because the pool of light varies more slowly than the contact shadow), converted as a radiance bake per light group with the dome black: it stores only the light the object's lamps send to the ground. Being drawn additively, each group (low beam, tail lamps) can be switched on or off at run time.

*References:* code: `apps/athenea/src/CmdMesh2Splat.cpp` (options `--light-catcher`, `--light-reach`); `docs/decisions.md`, "A car's lamps switched at run time: light groups (proposals 062, 066)" and its subsection "The ground: a light catcher"; `docs/operations.md` §3.6 ("A car's lamps").
:::

## 3.10 Validation against the path trace

`--validate DIR` measures a conversion material by material against the stage path-traced, for any stage. The groups
are the stage's material bindings (a mesh of two materials is in both); the conversion is the command's own, run
once per group with every mesh outside it hidden, so every option is what is measured. The ground truth is path
traced once (`--validate-paths` 512, `--validate-bounces` 6) and kept; the stage is rasterised once as meshes with
its Cryptomatte, and each group's mask is the coverage of every rank whose id hashes one of its meshes. Within the
mask's box the Measure effect gives relMSE, p99 and the means of the cloud and of the mesh raster beside it, written
to `validate.json`. `--validate-sky white` (or an image) draws everything under another sky with the stage's other
lights off, so a transfer's error is its occlusion's and reflection's alone. Among the defects it measures: a solid glass ball
traced at half the path-traced brightness (0.096 against 0.195) when rays are not bent into it, and a paint ball
under a white dome reading 13 % under because of the coat's Fresnel.

::: {.nota}
**`--validate`** — Automates the per-material measurement that was previously done with external scripts for a single asset.

- **Groups**: the stage's material bindings (`usd::stageMaterialGroups`), including those of GeomSubsets; a mesh of two materials is in both.
- **Conversion**: the command's own, once per group and with everything outside it hidden (`--hide`), so any option passed is the one measured.
- **Reference (GT)**: a path trace of the stage at 512 paths per pixel and 6 bounces, computed once and kept.
- **Mask**: the stage is rasterised as meshes with its Cryptomatte [47], which stores per pixel several (id, coverage) pairs sorted by coverage ("ranks"); the id is a hash of the prim's name. A group's mask is the sum of the coverages of the ranks whose id is one of its meshes': an antialiased mask, with no need for a frame per material.
- **Measurement**: the Measure effect (the same as `athenea compare`) over the mask's box: relMSE, p99 (the 99th percentile of the per-pixel error) and the means of the cloud and of the rasterised mesh, in `validate.json`.

**`--validate-sky white`** — Under a uniform white sky and with the other lights off (a "furnace"-type test), a transfer's error is only that of its occlusion and its reflection: there is no lighting structure to mask it.

**The defects cited** — A solid glass ball at half the brightness (0.096 against 0.195) when the bake's rays are not bent into it and do not see what the lens concentrates; a paint ball under a white dome 13 % under because of the coat's Fresnel (the coat reflects outwards part of the light that should have reached the base).

*References:* [47]; code: `apps/athenea/src/Mesh2SplatValidate.cpp` (`validateConversion`), `shaders/athenea/usd/m2s_validate.slang`, `modules/usd/src/MeshStage.cpp` (`stageMaterialGroups`); `docs/operations.md` §7.1 ("Measure"); `docs/decisions.md`, "A conversion measured by the engine itself" and "Measure: `athenea compare` as an AOFX effect".
:::

## 3.11 Measured results on the Corvette

All bakes below are TX transfers, `--cell-from-camera /World/Camera`, 64 paths per Gaussian, 3 bounces, no adaptive
extra paths, with the specular filter and edge reach at their defaults (1 and 1.5). "Car crop" is display PSNR on the
900 × 420 crop of the car, "frame" on the full 1920 × 1080 frame, both against the mesh path trace, after the merge
tree is cut at 0.25 px with back culling.

::: {.nota}
**The common settings** — *TX* is the transfer of §4; 64 paths per Gaussian and 3 bounces set the noise and the reach of the bake's indirect light; adaptive paths (more paths where the noise is high) are turned off so that comparisons do not depend on them.

**"Car crop" and "frame"** — Display PSNR is computed over the image transformed for display, on the $900\times420$ pixel crop of the car (which isolates the object's quality) and on the full $1920\times1080$ frame (which includes ground and sky, easier, and therefore gives higher PSNRs).

**The 0.25 px cut** — The cloud is drawn not with its leaves but with a cut of the merge tree (§6): the frontier of nodes whose projected screen error is below 0.25 pixels. With back-face culling, the nodes of closed shells facing away are also discarded (§3.5). It is the configuration used for playback.

*References:* §2.15 and §6 of the paper; `docs/operations.md` §2.4.1 (`--cut-bounds`, `--cut-cull-back`).
:::

**640 → 1280 px.** The 640 px bake's leaves were 3–4 mm on the hood, and the door's stripes and the sill's hairs —
absent from the path trace — were in the cloud. The car is re-baked at 1280 px in 8 groups of top-level parts
balanced by count, each converted with all other parts hidden (part names taken from the Cryptomatte manifest). The
eight bakes wrote 891 387, 896 496, 809 073, 790 403, 1 631 765, 1 073 954, 1 302 878 and 1 307 473 Gaussians
(8 703 429 in total) in 212–304 s of transfer each, about 34 min altogether.

::: {.nota}
**Leaves** — The conversion's Gaussians are the leaves of the merge tree; "leaves of 3–4 mm" is the size of a cell on the hood with `--camera-pixels 640`.

**Stripes and hairs** — Artefacts of the cloud absent from the path trace: stripes on the door and fine "hairs" (visible elongated Gaussians) on the sill. The hypothesis of §3.11 is that resolution reduces them; variants A/B/C test two other causes.

**Balanced groups** — Eight groups of top-level parts with similar counts, so that each fits in device memory and takes the same time. The part names come from the Cryptomatte manifest, the id → name table the renderer writes with the image.

**The sums** — $891\,387 + 896\,496 + 809\,073 + 790\,403 + 1\,631\,765 + 1\,073\,954 + 1\,302\,878 + 1\,307\,473 = 8\,703\,429$. Eight bakes of about 255 s on average give about 34 min.

*References:* [47]; §6 of the paper.
:::

| | 640 px | 1280 px |
|---|---|---|
| leaves | 5.38 M | 8.70 M |
| hood / door | 50 k / 79 k | 151 k / 266 k |
| glass, tyres | about one per triangle | about one per triangle |
| car crop / frame (PSNR) | 16.55 / 22.18 dB | 17.88 / 23.90 dB |

![Figure 7. Car crop (900 × 420 of the 1920 × 1080 frame from `/World/Camera`): path trace of the mesh (top), the 640 px bake (middle) and the 1280 px bake (bottom), both drawn as the 0.25 px cut with back-face culling. Car-crop PSNR 16.55 dB and 17.88 dB.](figures/img-car-640-1280.jpg)

![Figure 8. The full 1920 × 1080 frame of the 1280 px bake, 0.25 px cut with back-face culling (frame PSNR 23.90 dB against the path trace).](figures/img-car-1280-frame.jpg)

::: {.nota}
**Why it does not quadruple** — Doubling the camera's pixels halves the cell and would multiply by four the Gaussians of a surface resolved by cells. The hood goes from 50 k to 151 k (×3.0) and the door from 79 k to 266 k (×3.4), close to that factor; the total grows only ×1.62 (5.38 → 8.70 M) because much of the car (glass, tyres, small parts) was already at one Gaussian per triangle and gains nothing.

**PSNR** — +1.33 dB on the car crop and +1.72 dB on the frame: the MSE drops by 26 % and 33 % respectively.

*References:* §2.15 of the paper (image comparison on the GPU).
:::

Analysis of the 1280 px cloud against the 640 px one: hood highlights 3.2 → 2.0 px off their path-traced position,
the door's open-reflection contrast 0.35/0.53 → 0.56/0.74 of the path trace's, the grille's holes in the coarse cut
2.24 → 1.04, the sill's hairs 987 → 459 px from the resolution alone (Figures 6 and 7).

::: {.nota}
**The analysis's measurements** — Offset of the hood's highlights from their position in the path trace (3.2 → 2.0 px); contrast of the door's open reflections against the path trace's (0.35/0.53 → 0.56/0.74, the cloud's value over the reference's); holes in the grille in the coarse cut (2.24 → 1.04); and pixels covered by the sill's hairs (987 → 459 px).

**On the figure reference** — The parenthesis "(Figures 6 and 7)" refers to the images of this section; Figure 6 is the conversion diagram and the car images are Figures 7 and 8.

*References:* §6 of the paper (the merge tree cut).
:::

**Settings for 1280 px (variants A/B/C).** To select the 1280 px settings, eight parts (both doors, both sill panels, front fender,
chrome grille, grille rim, hood) were baked with the whole car casting, 931 017 Gaussians in each variant:

| variant | settings | result |
|---|---|---|
| **A** | as the 640 px bake ($k = 1$, $\rho = 1.5$) | selected: no hairs, the door smooth |
| B | `--specular-filter 0` | dashed dark line on the sill's fold; the chrome strip sparkles |
| C | B + `--edge-reach 1.0` | elongation and hairs identical to B |

![Figure 9. The 1280 px settings on the sill panel: the side of the car for context (top left), variant A (top right), variant B (bottom left) and variant C (bottom right). B and C show the dashed dark line on the fold.](figures/img-sill-variants.jpg)

The variants test whether the door's streaks are the coat's Toksvig widening saturating on folds and the sill's
hairs the edge pass's elongated Gaussians. Neither holds at 1280 px: the widening is needed on folds and the
tighter edge reach changes nothing, so the settings of A are used (Figure 9).

::: {.nota}
**Design of the variants** — Eight parts with the whole car casting shadow (so that the lighting is that of the complete car), 931 017 Gaussians in each variant: neither the specular filter nor the edge reach changes the count, so the differences are only of shape and roughness.

- **A** reproduces the 640 px settings.
- **B** removes the specular filter (hypothesis: the door's stripes are the coat's Toksvig widening saturating on the folds). Result: a dashed dark line appears on the sill's fold and the chrome strip sparkles; the widening was needed.
- **C** adds a tighter edge reach to B (hypothesis: the hairs are Gaussians elongated by the edge pass). Result: identical to B; the reach was not the cause.

The conclusion is that, at 1280 px, the resolution resolves what the hypotheses attributed to the settings.

*References:* §3.5 and §3.7 of the paper (edge reach and specular filter).
:::

**The 5120 px close-up level.** A close view stands far nearer than `/World/Camera`, about 0.5 mm per screen
pixel against 3–4 mm leaves. A logo-and-grille conversion from a close camera grows from 108 527 to
404 594 Gaussians between 1280 and 5120 px and shows the checkered flag, the bowtie and the grille's mesh crisp. The
66 parts the 1280 px bake had already given at most 1.5 Gaussians per triangle (badges, grilles, lamps, calipers,
mirrors, wheels, wipers, trims) are re-baked at 5120 px in seven groups; the panels at 5120 px or as fine as kept a
part under a cap of leaves (5 M, lowered to 4.5 M after a 5.01 M tree did not fit in device memory), with
`--max-cells 1048576` (Figure 10):

![Figure 10. The logo from a close camera: leaves of the 1280 px bake (left) and of the 5120 px close-up level (right).](figures/img-logo-1280-5120.jpg)

| groups | leaves | resolution (px) |
|---|---|---|
| fine0–6 (66 detailed parts) | 18 610 627 | 5120 |
| door | 4 084 170 | 5120 |
| Bottom | 4 897 428 | 3392 |
| Part_3 | 4 370 459 | 4480 |
| Infinite_Black | 4 481 738 | 4736 |
| six large panels and the windshield | 15 994 566 | 5120 |
| pan0–4 (small panels) | 17 088 152 | 5120 |
| Front_Bumper_001 | 5 008 986 | 4480 |
| **24 trees** | **74 536 126** | |

::: {.nota}
**Close-up level** — An additional level of detail for close views: at about 0.5 mm per screen pixel, a 3–4 mm leaf covers 6–8 pixels and is seen as a blot. 5120 px is four times 1280: a cell four times finer and up to sixteen times more Gaussians on a surface resolved by cells. The logo test grows ×3.7 (108 527 → 404 594) because many of its triangles were already below the cell.

**Which parts are redone** — The 66 parts with at most 1.5 Gaussians per triangle at 1280 px are the fine-detail ones (badges, grilles, lamps, calipers…): their triangles are smaller than the cell, the one-Gaussian-per-triangle of §3.4 carries the triangle's size and the mesh's interior detail is lost; at 5120 px the cell drops below the triangle and recovers it.

**The leaf cap** — A part's merge tree is built whole on the device; 5 M leaves was the planned cap, lowered to 4.5 M when a 5.01 M tree did not fit. Parts that would exceed the cap at 5120 px are converted at the finest resolution that respects it (3392, 4480 or 4736 px). `--max-cells 1048576` ($2^{20}$) avoids the holes of §3.3 at these resolutions.

**The table** — $18\,610\,627 + 4\,084\,170 + 4\,897\,428 + 4\,370\,459 + 4\,481\,738 + 15\,994\,566 + 17\,088\,152 + 5\,008\,986 = 74\,536\,126$. The 24 trees are 7 (fine0–6) + 1 (door) + 1 (Bottom) + 1 (Part_3) + 1 (Infinite_Black) + 7 (six panels and the windshield) + 5 (pan0–4) + 1 (bumper).

*References:* §6 of the paper; `docs/decisions.md`, "The Corvette on an iPhone (11-12 Oct)" and "A merge's second moment about its own centre (13 Oct)".
:::

The 24 merge trees hold 74 328 031 merges, 208 095 roots and 148 864 157 nodes in 49 634 pages (87.4 GB) and
share one occlusion codebook of 8 182 entries; Section 6.9 gives the details. (The conversion of Front_Bumper_001
emits 5 009 059 Gaussians; its tree holds 5 008 986 leaves, the figure used in the table.)

::: {.nota}
**The merge tree identities** — Each merge joins two nodes into one, so leaves − merges = roots: $74\,536\,126 - 74\,328\,031 = 208\,095$. The total nodes are leaves + merges: $74\,536\,126 + 74\,328\,031 = 148\,864\,157$. The 87.4 GB in 49 634 pages give about 1.76 MB per page and about 590 bytes per node (position, shape, harmonics, transfer and LOD metadata).

**The occlusion codebook** — A shared dictionary of 8 182 directional occlusion masks obtained by k-means [23]; each node stores an index instead of its mask (§6.9).

**5 009 059 against 5 008 986** — The 73 Gaussians of difference between the conversion and the tree's leaves are, in all likelihood, empty slots (zero opacity) that the tree build does not keep as leaves; the text does not detail it.

*References:* [23], [35]; §6 of the paper.
:::

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

::: {.nota}
**One normal and one cell** — A chrome tube a few pixels across turns its normal by almost $180^\circ$ across its width; Toksvig widening is isotropic and per triangle, and the sharp lobe is linear in the offset: neither reproduces a highlight that runs across the whole tube within a single Gaussian. The bumper's lower lip is a strip of triangles smaller than the cell (§3.4).

**Lamps without emission** — The transfer bake stores the response to external light; the lamps' own emission does not enter that bake (the car's lamps are handled as light groups, §3.9).

**Tinted thin glass** — See §3.8: a thin wall lets the background through untinted; until blending carries a coloured transmittance, tinted glass is converted as solid. One index per cloud except with layers, which give each Gaussian its specular index.

**The coverage mapping** — Assumes the sum of footprints is $2\pi\sigma^2$, that of a regular lattice (§3.4). The Gaussians of triangles smaller than a cell (own size), those that displacement splits (`m2sSplit`) and those displaced by relief overlap otherwise, so their coverage is not exactly the one requested.

**Instancers and velocities** — See §3.2. USD's `velocities` attribute (per-point velocity to extrapolate positions between samples and for motion blur) is not used: the position is read at the `--time` sample.

**Device memory** — About 5 M leaves per tree (4.5 M in practice, §3.11) on the M5 Pro.

*References:* `docs/decisions.md`, "A converted surface covers what its opacity says" (the note "The overlap `2 pi sigma^2` is the grid's").
:::
