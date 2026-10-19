<!-- Copyright (c) 2026 jesus luque. -->

# 10. References

References are given by authors, title and year as cited in the text. Where the sources this paper draws on name a
work without its full bibliographic details, it is listed as cited and marked so.

## 10.1 Gaussian splatting and point-based rendering

1. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field
   Rendering. ACM Transactions on Graphics (SIGGRAPH).
2. Kerbl, B., Meuleman, A., Kopanas, G., Wimmer, M., Lanvin, A., Drettakis, G. (2024). A Hierarchical 3D Gaussian
   Representation for Real-Time Rendering of Very Large Datasets. ACM Transactions on Graphics (SIGGRAPH).
3. Zwicker, M., Pfister, H., van Baar, J., Gross, M. (2001). EWA Volume Splatting / Surface Splatting. IEEE
   Visualization and SIGGRAPH; and (2002) EWA Splatting, IEEE Transactions on Visualization and Computer Graphics.
4. Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A. (2024). Mip-Splatting: Alias-free 3D Gaussian Splatting.
   CVPR.
5. Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S. (2024). 2D Gaussian Splatting for Geometrically Accurate Radiance
   Fields. SIGGRAPH.
6. Pfister, H., Zwicker, M., van Baar, J., Gross, M. (2000). Surfels: Surface Elements as Rendering Primitives.
   SIGGRAPH.
7. Moënne-Loccoz, N., et al. (2024). 3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes (3DGRT). ACM
   Transactions on Graphics (SIGGRAPH Asia).
8. Ren, K., et al. (2024). Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians.
9. Xiong, et al. (2026). NanoGS: pairwise mutual-nearest-neighbour merging of Gaussians. As cited in §6.2.
10. GaussianPOP: a per-Gaussian pruning score by the error its removal leaves. As cited in §2.5.2.
11. Spark: a 3D Gaussian splatting renderer for the web (World Labs). The player of §7 is a fork with a WebGPU backend.

::: {.nota}
**Notes on 10.1. Gaussian splatting and point-based rendering** — What each work contributed and where athenea uses it or departs from it. Code paths are relative to the public repository github.com/jesusluque/athenea.

**[1] Kerbl et al. 2023, 3D Gaussian Splatting (3DGS)** — Represents a scene as $N$ anisotropic 3D Gaussians, each with a mean $\mu$, a covariance $\Sigma = R S S^\top R^\top$ (a quaternion and log-scales), an opacity $\alpha$ and a colour in degree-3 spherical harmonics; it optimises them by differentiable rendering from photographs, with adaptive densification, and draws them with a tile-based rasteriser that sorts (tile, depth) keys and composites front to back, $C=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$. athenea takes the representation and the rasteriser family (§2.4–§2.5: `render::TileRasterizer`, `shaders/athenea/splat/splat_project.slang`, `splat_blend.slang`, the layout in `shaders/athenea/common/packing.slang`) and the clamping of the centre to the widened frustum. It departs in that it does not train: the Gaussians come from meshes (§3) and their light from a bake (§4); it splits the 64-bit sort into a depth sort of $N$ keys and a stable per-tile sort (§2.5); it cuts at $r^2_{\max}(\alpha)$ instead of at $3\sigma$; and it blends in linear light. DOI 10.1145/3592433; arXiv 2308.04079.

**[2] Kerbl et al. 2024, hierarchical 3D Gaussians** — Builds over a trained scene a hierarchy of nodes whose merged Gaussians come from weighted moments (weighted by opacity and area) and cuts it by on-screen size, with interpolation between levels and chunked training for very large scenes. athenea's original LOD (§6.1) follows those moments, $W=\sum w$, $S_1=\sum w\mu$, $S_2=\sum w(\Sigma+\mu\mu^\top)$ with $w=o\,A_2(s)$, but groups them in an octree by Morton code rather than in a hierarchy of its own (`shaders/athenea/lod/lod_common.slang`, which cites the work; `modules/lod/src/Lod.cpp`, `LodBuilder`). The merge trees of §6.2 preserve the sum of moments but merge in pairs and take the second moment about the group's centre. DOI 10.1145/3658160; arXiv 2406.12080.

**[3] Zwicker et al. 2001–2002, EWA splatting** — The elliptical weighted average filter projects a 3D Gaussian reconstruction kernel with the local affine approximation of the perspective (the Jacobian $J$ at the centre), which gives a 2D Gaussian $\Sigma' = J W \Sigma W^\top J^\top$, and convolves it with a screen-space low-pass filter to remove aliasing. The formula of §2.5.1, $\Sigma_{2D}=(JWM)(JWM)^\top$, is exactly this (`shaders/athenea/splat/frame.slang`, `splat_project.slang`; the reference in `shaders/athenea/reference/reference_project.slang`). The ray tracer of §2.6 avoids the approximation by evaluating each particle's peak along the ray, and §2.15 explains that the rasteriser matches it at p99 2 only where EWA is exact (orthographic views). DOI 10.1109/VISUAL.2001.964490 (EWA Volume Splatting), 10.1145/383259.383300 (Surface Splatting), 10.1109/TVCG.2002.1021576 (EWA Splatting).

**[4] Yu et al. 2024, Mip-Splatting** — Adds to 3DGS a 3D filter that limits each Gaussian's frequency to the maximum sampling frequency of the training views, and replaces the screen-space dilation with a 2D Mip filter that preserves energy by scaling the opacity by $\sqrt{\det\Sigma/\det(\Sigma+sI)}$. athenea belongs to that family (§1.5, §2.5.1): it dilates by 0.3 px² and scales the peak by $k=\sqrt{\det\Sigma_{\text{still}}/\det\Sigma'}$, a single square root that covers filter, shutter and lens; it also adds `paidOpacity`, which gives back the area lost to the cut at 1/255 (`shaders/athenea/splat/splat_project.slang`, `frame.slang`). It has no 3D filter: each Gaussian's size is set by the conversion's cell (§3.3). arXiv 2311.16493.

**[5] Huang et al. 2024, 2D Gaussian Splatting** — Flattens each Gaussian into an oriented disc (two tangent scales) and computes the exact ray–disc intersection as the crossing of the pixel's two homogeneous planes, with no affine approximation; an object-space low-pass filter takes the maximum of the disc and a screen-space Gaussian for edge-on views. In athenea a Gaussian with one scale exactly 0 is a surfel that the rasteriser draws this way, with the filter as a maximum against a round Gaussian of variance ½ px² and the opacity given back by area (§2.5.2; `shaders/athenea/splat/splat_project.slang`, `splat_blend.slang`, `modules/render/include/athenea/render/TileRasterizer.h`). The conversion writes them with `--flatness 0` (§3.4), and flat merges are written as surfels (`shaders/athenea/lod/lod_pairs_final.slang`, §6.2.1). It does not use the depth or normal regularisers, which belong to training. DOI 10.1145/3641519.3657428; arXiv 2403.17888.

**[6] Pfister et al. 2000, Surfels** — Defines the surfel (surface element) as a rendering primitive: a point with position, normal, radius and prefiltered attributes, drawn by visibility splatting over a z-buffer with hole filling. It is the conceptual antecedent of the disc of [5] and of the surface splatting of point clouds (§2.7, `modules/render/src/PointRasterizer.cpp`). DOI 10.1145/344779.344936.

**[7] Moënne-Loccoz et al. 2024, 3D Gaussian Ray Tracing (3DGRT)** — Traces Gaussian particles with the hardware BVH by wrapping each one in a proxy (a stretched icosahedron), collects the hits in a k-buffer sorted in segments along the ray and evaluates each particle at its maximum response, which enables secondary rays. athenea's *Hardware* route uses the same proxies (`shaders/athenea/rt/rt_proxy.slang`) with inline `RayQuery` (§2.6). It departs in the ordering: it records up to 256 unsorted entries and sorts them after the traversal with a counting sort into 128 buckets plus insertion (from 2.52 s to 343 ms against the 16-entry k-buffer), and on Metal `Auto` picks the ComputeBvh route (Karras's LBVH) because every non-opaque candidate leaves the hardware traversal (`modules/render/src/GaussianRayTracer.cpp`, `shaders/athenea/rt/rt_integrate.slang`). DOI 10.1145/3687934; arXiv 2407.07090.

**[8] Ren et al. 2024, Octree-GS** — Organises neural anchors (in the style of Scaffold-GS) in the levels of an octree and picks each anchor's level by view distance, with progressive level-by-level training. It is cited as related work (§1.5); athenea does not use it: its octree (§6.1) groups explicit Gaussians by Morton prefixes and merges them by moments, with no anchors or MLPs, and its production LOD is the merge-tree one (§6.2). arXiv 2403.17898.

**[9] Xiong et al. 2026, NanoGS** — Training-free simplification that merges in pairs Gaussians that are mutual nearest neighbours under a cost, iteratively. `athenea decimate --ratio`/`--tree` runs it on the device (§6.2): each node weighs 8 neighbours on each side along two Morton orders (one shifted by half a cell), pairs that choose each other are merged under a cost threshold chosen by histogram, up to half the nodes per pass, and piece, glass, mirror, fold and colour guards (§6.2.3) and the relaxed top (§6.2.4) are added. Code: `shaders/athenea/lod/lod_pairs.slang`, `lod_pairs_final.slang`, `modules/lod/src/Lod.cpp`, `apps/athenea/src/CmdDecimate.cpp`. No full bibliographic details in the sources; no identifier is given.

**[10] GaussianPOP** — A pruning score that measures, per Gaussian, the squared error its removal leaves in the views. athenea accumulates it in the same traversal as the blend, together with the contribution $T\alpha$ (§2.5.2, `shaders/athenea/splat/splat_blend.slang`), and `athenea decimate --prune-error` uses it (`modules/usd/src/Decimate.cpp`); the manual (`docs/operations.md` §2.4.1) records that on the Corvette's pieces it leaves holes in opaque surfaces, so it is kept for clouds that are not surfaces. No full bibliographic details; no identifier is given.

**[11] Spark (World Labs)** — A 3DGS renderer for the web on three.js, with its own LoD (`LodSplat`) and a "harder kernel": an opacity $D>1$ is drawn as $1-(1-g)^A$ with $A=\exp((D^2-1)/e)$. athenea adopts that kernel for merges with opacity above 1 (§2.5.2; `shaders/athenea/splat/frame.slang`, `splat_project.slang`, `lod_pairs_final.slang`, `--merge-opacity falloff` of `athenea decimate`). The player of §7 is the sparkwebGPU fork with a WebGPU backend (public demos at https://sparkwebgpu.lucab.co.uk). URL github.com/sparkjsdev/spark.
:::

## 10.2 Meshes to Gaussians

12. Electronic Arts (SEED). mesh2splat: converting meshes into 3D Gaussian splats by rasterisation (open source,
    BSD-3). The origin of the conversion of §3.
13. Mesh2GS. arXiv 2606.21898 (2026). The camera-derived cell size of §3.3.

::: {.nota}
**Notes on 10.2. Meshes to Gaussians** — The two works that set the shape and size of each converted Gaussian.

**[12] Electronic Arts (SEED), mesh2splat** — For each triangle, a geometry shader projects it onto the axis plane its normal points to most (“triplanar” coordinates), rasterises it at a fixed resolution, and each covered texel emits a Gaussian lying in the triangle's plane, sized by the Jacobian of the inverse projection times $\sigma=0.65$ and $1/N$, with a thickness of $10^{-7}$ units, appended with an atomic counter. athenea starts from it (§3.1, §3.4) but rewrites it as a compute kernel inside an AOFX effect (`plugins/mesh2splat/mesh2splat.slang`, `plugins/mesh2splat/Mesh2Splat.cpp`) with deterministic emission in three passes (count, scan, write), $\sigma=1.0$ by default, thickness as a fraction $\varphi$, opacity as coverage, the bound $\ell_{\max}$, `m2sOwnSize` for triangles smaller than a cell, hard edges (`m2sHoldEdges`), Toksvig (`m2sSpread`) and layered OpenPBR capture. URL github.com/electronicarts/mesh2splat.

**[13] Mesh2GS** — Derives a Gaussian's size from the footprint of one pixel of a camera, $h = z\,(A/P)/f$, with $z$ the minimum distance to the box inside the frustum and over the camera's time samples. athenea follows it with `--cell-from-camera CAM --camera-pixels P` (§3.3), but takes the Euclidean distance from the eye to the box bounded at `--time`, with a minimum of $z_{\text{near}}$: the same for a visible box, and finer, never coarser, for one beside the frustum. Code: `shaders/athenea/usd/mesh2splat_cells.slang` and `apps/athenea/src/CmdMesh2Splat.cpp`. arXiv 2606.21898 (as the paper cites it).
:::

## 10.3 Light transport, precomputed transfer and compression

14. Sloan, P.-P., Kautz, J., Snyder, J. (2002). Precomputed Radiance Transfer for Real-Time Rendering in Dynamic,
    Low-Frequency Lighting Environments. SIGGRAPH.
15. Sloan, P.-P., Hall, J., Hart, J., Snyder, J. (2003). Clustered Principal Components for Precomputed Radiance
    Transfer. SIGGRAPH.
16. Sloan, P.-P. (2008). Stupid Spherical Harmonics (SH) Tricks. Game Developers Conference. (Zonal harmonic fitting.)
17. Ramamoorthi, R., Hanrahan, P. (2001). An Efficient Representation for Irradiance Environment Maps. SIGGRAPH.
18. Křivánek, J., Colbert, M. (2008). Real-time Shading with Filtered Importance Sampling. Computer Graphics Forum
    (EGSR).
19. Lagarde, S., de Rousiers, C. (2014). Moving Frostbite to Physically Based Rendering. SIGGRAPH course notes.
20. Toksvig, M. (2005). Mipmapping Normal Maps. Journal of Graphics Tools.
21. Veach, E., Guibas, L. (1995). Optimally Combining Sampling Techniques for Monte Carlo Rendering. SIGGRAPH. (The
    power heuristic of §2.11.)
22. Conty Estevez, A., Kulla, C. (2018). Importance Sampling of Many Lights with Adaptive Tree Splitting. HPG. (The
    light BVH of §2.10.)
23. Lloyd, S. (1982). Least Squares Quantization in PCM. IEEE Transactions on Information Theory; Arthur, D.,
    Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. SODA. (The k-means fits of §7.2–7.3.)

::: {.nota}
**Notes on 10.3. Light transport, precomputed transfer and compression** — The works the core of the baker (§4) and the codecs of Part II (§7) rest on.

**[14] Sloan, Kautz and Snyder 2002, Precomputed Radiance Transfer (PRT)** — Precomputes per point a transfer vector, the spherical-harmonic projection of visibility times the cosine (and, in its interreflected version, of the bounced light), $T_k=\frac1\pi\int V(\omega)\max(n\cdot\omega,0)Y_k(\omega)\,d\omega$; with the environment projected into $L_k$, diffuse radiance is the dot product $\sum_k T_k L_k$, and the specular part needs a transfer matrix. athenea bakes exactly that $T_k$ with the path tracer in its `kTransfer` variant (§4.4.1, `modules/technique/src/PathTracer.cpp`, the variant declared in `modules/technique/include/athenea/technique/PathTracer.h`), with the direct and indirect halves kept apart and degree 3 (16 coefficients); it reads it in `shaders/athenea/splat/splat_relight.slang`. It departs on the specular side: it rejects the 9 × 9 transfer matrix per Gaussian (§4.6.2) and uses the reflected field and the open directions instead. DOI 10.1145/566654.566612.

**[15] Sloan et al. 2003, Clustered PCA (CPCA)** — Clusters the transfer vectors (vector quantisation) and applies PCA within each cluster, $t\approx\mu_c+\sum_j w_j d_{c,j}$; at draw time the light is projected once per cluster onto its basis, and each point only takes a dot product with its weights. athenea uses it twice (§7.2): the file's encoding 3 (`athc-convert --cpca`, a tool of the sparkwebGPU player) and the joint basis resident on the phone's GPU, whose "projected route" in the player is Sloan's per-cluster projection. In the engine, `shaders/athenea/lod/transfer_codec.slang` (which cites the work) and `athenea decimate --transfer-basis`. DOI 10.1145/882262.882281.

**[16] Sloan 2008, Stupid Spherical Harmonics (SH) Tricks** — GDC notes with, among other things, zonal harmonics: a function symmetric about an axis $a$ is written $g_{lm}=z_l\,k_l\,Y_{lm}(a)$, $k_l=\sqrt{4\pi/(2l+1)}$, and rotates by rotating its axis, with no per-band rotation matrix; and the fitting of zonal lobes to an SH function. athenea stores two zonal lobes per Gaussian in its own frame (§4.4.5, §5.7), with $z_l=k_l\sum_m f_{lm}Y_{lm}(a)$ and residual $\lVert f\rVert^2-\sum_l z_l^2$, fitted in the AOFX effect `plugins/splattransferzonal/splattransferzonal.slang` and read by `splatTransferFrame`.

**[17] Ramamoorthi and Hanrahan 2001** — Show that irradiance is the convolution of the environment with the clamped cosine, which is diagonal in SH, $E(n)=\sum_{lm}\hat A_l L_{lm}Y_{lm}(n)$ with $\hat A_0=\pi$, $\hat A_1=2\pi/3$, $\hat A_2=\pi/4$, and that since $\hat A_l$ decays fast nine coefficients represent it with little error. athenea reads the irradiance of a body without transfer this way (§4.3, `shaders/athenea/technique/env_project.slang`, `shaders/athenea/light/environment.slang`, both of which cite the work) and uses the same normalised weights $\{1, 2/3, 1/4\}$ in the quotient $O_{\text{SH}}$ of §4.5. DOI 10.1145/383259.383317.

**[18] Křivánek and Colbert 2008, filtered importance sampling** — To prefilter an environment with few samples, each BRDF sample reads the map at the mip level whose texel has the sample's solid angle, $\Omega_s=1/(Np)$, which removes the noise without more samples. athenea applies it when building the eight-level octahedral prefiltered map (§4.3, `shaders/athenea/technique/env_prefilter.slang`): each GGX sample reads a level coarser than its solid angle asks for and never narrower than the texel it writes, with 64 to 1024 samples per level. DOI 10.1111/j.1467-8659.2008.01253.x.

**[19] Lagarde and de Rousiers 2014, Moving Frostbite to PBR** — Course notes that include an approximation of specular occlusion from ambient occlusion, $\operatorname{sat}((n\cdot\omega_o+o)^{2^{-16r-1}}-1+o)$. athenea's first transfer used it with $o$ taken from $T_0$ (§4.6.1); TX replaces it by asking the lobe itself about its cells (`splatLobeOpen`, `splatConeOpen`) and keeps it only as the opening of the polish of smooth materials ($\chi=0$, §4.6.3). Code: `shaders/athenea/splat/splat_relight.slang`.

**[20] Toksvig 2005, Mipmapping Normal Maps** — The mean length of averaged unit normals, $|\bar N|<1$, measures their spread, and the specular lobe widens accordingly; in microfacet terms, the variance of the normals adds to that of the slopes. athenea applies the idea in the conversion but measures the spread geometrically: the turning rate of the vertex normals times the Gaussian's size, $s$, and $\alpha'^2=\alpha^2+2s^2$ (§3.7, §4.9.1; `m2sSpread`/`m2sWidened` in `plugins/mesh2splat/mesh2splat.slang`). At draw time it undoes it for the sharp lobes (`unwidened`, §4.9.2; `shaders/athenea/splat/splat_project.slang`, `splat_relight.slang`).

**[21] Veach and Guibas 1995, multiple importance sampling** — Combining several sampling strategies with weights $w_i=p_i^\beta/\sum_j p_j^\beta$ gives an unbiased estimator with near-optimal variance; with $\beta=2$ it is the power heuristic. athenea's path tracer weighs NEE against material sampling this way, $w_{\text{light}}=p_l^2/(p_l^2+p_m^2)$ with $p_l=P(k\mid x)\,p_k(\omega)$, and leaves weight 1 to delta lights and lobes (§2.11). The path tracing kernel is generated by `modules/technique/src/PathTracer.cpp`; the white furnace uncovered a one-sided weight. DOI 10.1145/218380.218498.

**[22] Conty Estevez and Kulla 2018, many lights** — A light BVH whose nodes store a box, an orientation cone and a power; a node's importance for a point bounds distance and angle, and a stochastic traversal picks a light with known probability. athenea builds it with the same LBVH kernels as the geometry and uses it with `athenea:chooseLights` (§2.10; `shaders/athenea/light/light_bvh.slang`, which cites the cone-union algorithm 1, and `light_bvh_build.slang`); its probability enters the MIS as $P(k\mid x)$. The paper does not describe the work's adaptive tree splitting. DOI 10.1145/3233305.

**[23] Lloyd 1982; Arthur and Vassilvitskii 2007, k-means and k-means++** — Lloyd alternates assigning each point to the nearest centroid and recomputing each centroid as a mean, which never increases the distortion; k-means++ seeds by sampling with probability proportional to $D^2$, with an $O(\log k)$ guarantee in expectation. athenea uses them in `fit-compact-basis` (k-means++ seeds, Lloyd over all nodes, then PCA per cluster; a tool of the sparkwebGPU player, §7.2.2) and in the open-directions dictionary, a k-means in Hamming space whose centroid is the bitwise majority (`shaders/athenea/lod/shadow_vq.slang`, `athc-convert --shadow-vq`, §7.3). The occlusion codebook of §6.7 is a variant with the containment constraint (`shaders/athenea/lod/occlusion_mask.slang`). DOI (Lloyd) 10.1109/TIT.1982.1056489.
:::

## 10.4 Materials and shading

24. Walter, B., Marschner, S., Li, H., Torrance, K. (2007). Microfacet Models for Refraction through Rough Surfaces.
    EGSR.
25. Heitz, E. (2018). Sampling the GGX Distribution of Visible Normals. Journal of Computer Graphics Techniques.
26. Oren, M., Nayar, S. (1994). Generalization of Lambert's Reflectance Model. SIGGRAPH; Portsmouth, J., Kutz, P.,
    Hill, S. (2025). EON: A Practical Energy-Preserving Rough Diffuse BRDF.
27. Burley, B. (2012). Physically-Based Shading at Disney. SIGGRAPH course notes.
28. Estevez, A. C., Kulla, C. (2017). Production Friendly Microfacet Sheen BRDF. Sony Pictures Imageworks;
    Zeltner, T., Burley, B., Chiang, M. J.-Y. (2022). Practical Multiple-Scattering Sheen Using Linearly Transformed
    Cosines.
29. Lazarov, D. (2013). Getting More Physical in Call of Duty: Black Ops II. SIGGRAPH course notes. (The environment
    albedo fit compared in §4.6.4.)
30. Andersson, Z., Kutz, P., et al. (2024). OpenPBR Surface specification. Academy Software Foundation.
31. MaterialX: an open standard for rich material and look-development content (version 1.39). Academy Software
    Foundation.

::: {.nota}
**Notes on 10.4. Materials and shading** — The lobe library (`shaders/athenea/material/lobes.slang`, `microfacet.slang`) and the material standards the conversion reads.

**[24] Walter et al. 2007, microfacets with refraction** — Gives the BTDF of a rough dielectric with the generalised half vector $h_t=-(\eta_i\,\omega_i+\eta_o\,\omega_o)$, $f_t=\frac{|\omega_i\cdot h||\omega_o\cdot h|}{|\omega_i\cdot n||\omega_o\cdot n|}\frac{\eta_o^2(1-F)GD}{(\eta_i(\omega_i\cdot h)+\eta_o(\omega_o\cdot h))^2}$, and introduces the GGX distribution. athenea's transmission follows Walter (§2.9): `transmissionNormal`, `microfacetTransmission` and `transmissionPdf` in `shaders/athenea/material/lobes.slang`, with each face's Fresnel paid by the lobe itself; GGX with $\alpha=r^2$ is the distribution of the whole engine (§4.1). DOI 10.2312/EGWR/EGSR07/195-206.

**[25] Heitz 2018, visible normal sampling (VNDF)** — Samples the distribution of normals visible from $\omega_o$, $D_{\omega_o}(h)=G_1(\omega_o)\max(0,\omega_o\cdot h)D(h)/\cos\theta_o$, by stretching the problem to the isotropic roughness-1 configuration and sampling a projected disc; it reduces variance compared with sampling $D$. athenea's microfacet lobes sample this distribution (§2.9) and their density is Heitz's (`ggxVisibleNormalPdf`), but the sampling uses the spherical-cap form of Dupuy and Benyoub 2023 (`ggxSampleVisibleNormal` in `shaders/athenea/material/microfacet.slang`), which gives the same distribution with less arithmetic. JCGT 7(4), jcgt.org/published/0007/04/01.

**[26] Oren and Nayar 1994; Portsmouth, Kutz and Hill 2025 (EON)** — Oren–Nayar models a rough diffuse surface as Lambertian V-cavities with slope deviation $\sigma$, which darkens the centre and brightens the rim compared with Lambert; EON is OpenPBR's energy-preserving variant with a multiple-scattering term and exact albedo and sampling. athenea has both: `kLobeOrenNayar` with `kFlagEnergyCompensation` calls `orenNayarCompensated` (`shaders/athenea/material/lobes.slang`), described as "OpenPBR's energy-compensated Oren-Nayar (EON)" in `microfacet.slang` (§2.9). DOI (Oren–Nayar) 10.1145/192161.192213.

**[27] Burley 2012, Physically-Based Shading at Disney** — The "principled" model: a diffuse term with retroreflection $f_d=\frac\rho\pi(1+(F_{D90}-1)(1-\cos\theta_l)^5)(1+(F_{D90}-1)(1-\cos\theta_v)^5)$, $F_{D90}=0.5+2r\cos^2\theta_d$, GGX, Schlick and the $\alpha=r^2$ convention for perceptual roughness. athenea has the `kLobeBurley` lobe (`shaders/athenea/material/lobes.slang`) and the MaterialX node `shaders/athenea/material/mx/athenea_burley_diffuse_bsdf.slang`; the $\alpha=r^2$ convention is that of all of §4.

**[28] Estevez and Kulla 2017; Zeltner, Burley and Chiang 2022, sheen** — Imageworks proposes a microfibre distribution $D(m)=(2+1/r)\sin^{1/r}\theta/(2\pi)$ for the sheen of fabrics ("Charlie sheen"); Zeltner et al. fit a volumetric sheen with multiple scattering using linearly transformed cosines, with cheap evaluation, sampling and albedo. The library has both forms (`imageworksSheenD`, `imageworksSheenDirectionalAlbedo`, `zeltnerSheenDirectionalAlbedo` in `shaders/athenea/material/microfacet.slang`; `mx/athenea_sheen_bsdf.slang`). In the relight, the sheen responds as a body, its directional albedo times $\mathbf B$ under the coat (§4.6.4, `splat_relight.slang`).

**[29] Lazarov 2013, Black Ops II** — A cheap analytic fit of the environment term of the split sum (the directional albedo of the specular BRDF, DFG). athenea used it and replaced it (§4.6.4): at grazing angles it differed by up to 76 % from MaterialX's `ggxDirectionalAlbedo` with multiple-scattering compensation, which is what is used now (`shaders/athenea/splat/splat_relight.slang`; the check in `shaders/athenea/test/dfg_check.slang`).

**[30] Andersson, Kutz et al. 2024, OpenPBR Surface** — The ASWF's layered uber-shader (the merger of Autodesk Standard Surface and Adobe Standard Material): base (metal as generalised Schlick with F82 tint, dielectric, EON diffuse, subsurface, transmission), specular, coat with base darkening, fuzz (sheen), thin film, emission, `geometry_opacity` and `geometry_thin_walled`. The conversion reads `open_pbr_surface` with its names and defaults (§3.2, `modules/usd/src/MeshStage.cpp`), captures its layers per Gaussian (§3.6), writes the Schlick metal (`schlickMetal`), and the relight applies the coat darkening in $U$ (§4.6.4). Compiled to Slang it takes 945 lines and 55 blob words (§2.9). URL github.com/AcademySoftwareFoundation/OpenPBR.

**[31] MaterialX 1.39** — A node-graph standard with standard libraries (pbrlib: BSDF nodes, `layer`, `mix`) and shader generators (GLSL, MSL, OSL, MDL and, since 1.39, Slang). athenea's `MaterialCompiler` derives from the `SlangShaderGenerator` of MaterialX 1.39.5 and changes the meaning of four node classes (surface, BSDF/EDF, image, `heighttonormal`); it is checked against MaterialX's genglsl to $3.2\cdot10^{-5}$ (§2.9). Code: `modules/material/src/MaterialCompiler.cpp`, `modules/technique/src/MaterialPrograms.cpp`, `shaders/athenea/material/mx/`. URL materialx.org.
:::

## 10.5 Geometry, acceleration structures and levels of detail

32. Karras, T. (2012). Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees. High Performance
    Graphics.
33. Möller, T., Trumbore, B. (1997). Fast, Minimum Storage Ray–Triangle Intersection. Journal of Graphics Tools.
34. Runnalls, A. R. (2007). Kullback–Leibler Approach to Gaussian Mixture Reduction. IEEE Transactions on Aerospace
    and Electronic Systems.
35. Karis, B., Stubbe, R., Wihlidal, G. (2021). Nanite: A Deep Dive. SIGGRAPH Advances in Real-Time Rendering.

::: {.nota}
**Notes on 10.5. Geometry, acceleration structures and levels of detail** — The four algorithmic pieces reused by mesh visibility, the ray tracer and the merge trees.

**[32] Karras 2012, parallel BVH construction** — Over sorted Morton keys, each internal node $i$ of a radix tree decides its direction by the sign of $\delta(i,i+1)-\delta(i,i-1)$ ($\delta$ the longest common prefix), finds the other end of its range by binary search and splits at the highest differing bit; all nodes in parallel, $O(n)$, and the boxes are propagated upwards with atomic counters. It is the LBVH of the Gaussian ray tracer's ComputeBvh route (§2.6; `shaders/athenea/rt/bvh_hierarchy.slang`, `bvh_common.slang`, level-by-level refit in `bvh_refit_level.slang`), of `VisibilityBvh` per mesh and over the instances (§2.8; `modules/world/src/BvhScene.cpp`) and of the light BVH (§2.10). DOI 10.2312/EGGH/HPG12/033-037.

**[33] Möller and Trumbore 1997, ray–triangle intersection** — Solves $o+td=(1-u-v)a+ub+vc$ by Cramer's rule with cross products ($P=d\times e_2$, $\det=e_1\cdot P$, $u=T\cdot P/\det$, $Q=T\times e_1$, $v=d\cdot Q/\det$, $t=e_2\cdot Q/\det$), without storing the plane equation. athenea rebuilds the hit from the visibility buffer with it, in view space, so that the three routes shade alike (§2.8; `shaders/athenea/technique/surface.slang`), and uses it in the traversal of `shaders/athenea/technique/visibility_bvh.slang`. DOI 10.1080/10867651.1997.10487468.

**[34] Runnalls 2007, Gaussian mixture reduction** — Merges pairs of components by moments and picks the pair with the smallest upper bound on the Kullback–Leibler divergence, $B=\tfrac12[(w_i+w_j)\log\det\Sigma_{ij}-w_i\log\det\Sigma_i-w_j\log\det\Sigma_j]$. The shape term of athenea's merge cost is that bound per unit mass (§6.2.2), with the second invariant $I_2$ (the sum of pairwise products of eigenvalues) instead of the determinant, so that a surfel with no volume has a finite cost (`shaders/athenea/lod/lod_pairs.slang`, which cites the work). The player uses the same cost in `athc-convert --error`. DOI 10.1109/TAES.2007.4383588.

**[35] Karis, Stubbe and Wihlidal 2021, Nanite** — In a DAG of simplified clusters, each group carries an error and a sphere forced to be monotonic (the parent with error greater than or equal to its children's and with a sphere containing theirs), so that the cut is decided per cluster in parallel with the test "my projected error $\le\tau<$ my parent's"; streaming pages are ordered by that error. athenea applies exactly that construction to the Gaussian merge trees (§6.3, the proposition of a leaf drawn exactly once; `treeBoundsUp`, `treeErrorKeep` in `shaders/athenea/lod/lod_tree_cut.slang` and `modules/usd/src/Decimate.cpp`) and pages by bands of the parent's error (§6.8, `shaders/athenea/lod/lod_pages.slang`; in the player, §7.5.1). URL advances.realtimerendering.com/s2021.
:::

## 10.6 Colour

36. Academy of Motion Picture Arts and Sciences (2024). ACES 2.0 output transforms (CTL reference).
37. Hellwig, L., Stolitzka, D., Fairchild, M. D. (2022). The colour appearance model known as Hellwig 2022, a revision
    of CAM16, used by the ACES 2.0 output transforms.
38. Wrensch, B. An analytic fit of the AgX view transform. Online note, as cited in §2.12.
39. Krystek, M. (1985). An Algorithm to Calculate Correlated Colour Temperature. Color Research and Application. (The
    blackbody fit of §2.10.)
40. OpenColorIO. Academy Software Foundation.

::: {.nota}
**Notes on 10.6. Colour** — The display (`technique::DisplayTransform`) and the colour of lights.

**[36] AMPAS 2024, ACES 2.0 output transforms** — They replace the RRT/ODT of ACES 1 with a single transform parametrised by the display's peak luminance: a tone scale, chroma compression and a gamut compression done in the space of a colour appearance model ([37]), with its reference in CTL. athenea ports it function by function to Slang (§2.12; `shaders/athenea/technique/aces2.slang`), with the tables the reference computes at start-up (362 entries per hue) built on the device by `aces2_prepare.slang` and orchestrated by `modules/technique/src/Aces2.cpp`; greys stay neutral to $5\cdot10^{-6}$ and the round trip holds to $6\cdot10^{-6}$ of the peak.

**[37] Hellwig, Stolitzka and Fairchild 2022** — A revision of CAM16 that simplifies the model and corrects its colourfulness and chroma, giving the appearance coordinates $J$ (lightness), $M$ (colourfulness) and $h$ (hue). ACES 2.0 does its gamut compression in that JMh space, and `aces2.slang` implements it "as the reference specialises it" (§2.12).

**[38] Wrensch, Minimal AgX** — AgX is Troy Sobotka's view transform; Wrensch published a minimal analytic fit: an input matrix, a log2 clamped to $[-12.47, 4.03]$ stops and a sixth-order polynomial sigmoid. athenea offers it as view 1 of `shaders/athenea/technique/display.slang`, which cites the 2023 fit (§2.12).

**[39] Krystek 1985, colour temperature** — A rational fit of the Planckian locus in CIE 1960 $(u,v)$ coordinates as a function of temperature, valid over the usual range of lamps. athenea uses it for the blackbody of a UsdLux light's colour temperature, converted to linear sRGB and normalised to luminance 1 (§2.10; `shaders/athenea/light/lights.slang`, "Krystek's rational fit of the Planckian locus").

**[40] OpenColorIO (ASWF)** — A colour management library and configuration format; its GPU path generates a shader's text (GLSL, HLSL, MSL) and the values of its LUTs. athenea uses it as a compiler (§2.12): `colour::ColourCompiler` asks for the HLSL, loads it as a Slang module `athenea_cs_<hash>` and fills the LUTs on the device (`modules/colour/src/ColourCompiler.cpp`, `modules/technique/src/Ocio.cpp`); `colour::ColourNames` resolves the input colour spaces of textures (§2.9, `modules/colour/src/ColourNames.cpp`). URL opencolorio.org.
:::

## 10.7 Animation and visibility

41. Lin, et al. Directional visibility MLPs per body part for articulated avatars. As cited in §5.8.
42. DNF-Avatar: a scalar ambient-occlusion grid per bone. As cited in §5.8.

::: {.nota}
**Notes on 10.7. Animation and visibility** — The two ideas that the per-part directional table of §5.8 combines. Neither has full bibliographic details in the sources, so no identifier is given.

**[41] Lin et al., per-part visibility MLPs** — They factor the visibility of an articulated avatar by body part, with a directional network per part that moves rigidly with it, so that the total occlusion is a product that survives the pose. athenea takes the factorisation and replaces the network with a table: per part, a grid of $N^3$ probes each with an octahedral map of $O^2$ transmittances, baked once in the bind pose and read per frame as $V_{i\ell}=\prod_{p\ne\pi(i)}F_p(T_p^{-1}x_i, L(T_p)^{-1}\omega_\ell)$. Code: `shaders/athenea/splat/splat_visibility.slang` (which cites Lin et al. 2024), `splat_visibility_bake.slang`, `splat_visibility_read.slang`, `modules/technique/src/SplatVisibility.cpp` (`partitionJoints`, `visPartFrames`), CLI `athenea visibility`.

**[42] DNF-Avatar** — Bakes ambient occlusion as a scalar grid per bone. athenea keeps from it the ambient term of each probe (the transmittance averaged over directions), which is what is used for domes (§5.8, same files); the direction comes from [41].
:::

## 10.8 Systems and standards

43. Pixar Animation Studios. OpenUSD (Universal Scene Description) and Hydra, including UsdSkel, UsdLux, UsdShade and
    the `ParticleField` schema.
44. He, Y., Fatahalian, K., Foley, T. (2018). Slang: Language Mechanisms for Extensible Real-time Shading Systems. ACM
    Transactions on Graphics; slang-rhi, the rendering hardware interface used by the engine.
45. Intel. Open Image Denoise (OIDN) 2.5.
46. W3C. WebGPU and the WebGPU Shading Language (WGSL).
47. Friedman, J., Jones, A. C. (2015). Fully Automatic ID Mattes with Support for Motion Blur and Transparency
    (Cryptomatte). SIGGRAPH Posters.
48. O'Neill, M. E. (2014). PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number
    Generation; Halton, J. H. (1960). On the efficiency of certain quasi-random sequences of points.

::: {.nota}
**Notes on 10.8. Systems and standards** — The infrastructure everything runs on, and where it hooks in.

**[43] OpenUSD and Hydra (Pixar)** — USD composes a scene from layers, references, payloads and variant sets, with schemas such as UsdGeom, UsdShade (material bindings, `GeomSubset` with the `materialBind` family), UsdLux (lights), UsdSkel (skeletons, `SkelBindingAPI`, `UsdSkelBakeSkinning`) and `ParticleField`, whose `ParticleField3DGaussianSplat` is the master's prim. Hydra is the rendering architecture: a render delegate receives rprims, sprims and bprims, and scene indices transform the scene on the way. athenea is the `hdAthenea` delegate (`modules/usd/src/RenderDelegate.cpp`; §2.13), reads `ParticleField` in `modules/usd/src/ParticleField.cpp`, adds the scene index `HdAtheneaSkelSplatSceneIndex` (`modules/usd/src/SkelSplat.cpp`, §5.5), and the conversion reads the stage directly with UsdShade (`modules/usd/src/MeshStage.cpp`, §3.2). URL openusd.org.

**[44] He, Fatahalian and Foley 2018, Slang; slang-rhi** — Slang extends HLSL with modules, generics, interfaces, link-time specialisation (`extern static const`) and a reflection API, and compiles to SPIR-V, Metal, CUDA and WGSL; slang-rhi is its rendering hardware layer over Metal, Vulkan, CUDA and WebGPU. Every athenea kernel is Slang (§2.1), parameters are bound by name through reflection, link-time constants give one program per set of values, and `DiskShaderCache` keeps one file per program (§2.3; `modules/gpu`). The sparkwebGPU player recompiles athenea kernels to WGSL (§5.12, §7.8). DOI 10.1145/3197517.3201380.

**[45] Intel Open Image Denoise 2.5** — A convolutional-network (U-Net) denoiser for path-traced images, optionally guided by albedo and normal, with GPU devices. `technique::Denoiser` runs its `RT` filter on the engine's queue (§2.11; `modules/technique/src/Denoiser.cpp`), built for GPU only (rule 5 of §2.1), on Metal through intermediate buffers with hazard tracking and on Vulkan through CUDA over imported memory. URL openimagedenoise.org.

**[46] W3C, WebGPU and WGSL** — The web's GPU API and its shading language, with limits that settle several of the player's decisions: ten storage buffers per stage in Chrome on a Mac (which is why the player rewrites `splatSkin` in a kernel of its own, §5.12), the workgroup memory of the sharp lobe's slope (§4.8.2), and the ban on binding the same buffer as read and write in one dispatch (§7.8). URLs w3.org/TR/webgpu and w3.org/TR/WGSL.

**[47] Friedman and Jones 2015, Cryptomatte** — Automatic ID mattes: each object name is hashed (32-bit MurmurHash3, with the exponent fixed so that the id is a valid float), each pixel stores (id, coverage) pairs sorted by coverage in RGBA layers of two ranks, and a manifest in the metadata maps names to ids. athenea computes the id in `modules/core/src/Hash.cpp` (`murmurHash3x86_32`, `cryptomatteFixup`, `cryptomatteId`), accumulates six ranks in the blend (§2.5.2), stores `cryptoObject` per Gaussian and `cryptoManifest` per cloud, and uses it for the masks of `--validate` (§3.10), the groups of `--object-groups` (§6.2.5) and the `CryptoObject00`–`02` layers of the OpenFX node (§2.14). URL github.com/Psyop/Cryptomatte.

**[48] O'Neill 2014, PCG; Halton 1960** — PCG advances a linear congruential generator and applies a permutation to the output (xorshift and a state-dependent rotation), with good statistical quality and a small state; the Halton sequence is the radical inverse in prime bases, a low-discrepancy sequence. The path tracer's sampler folds pixel, stream, absolute path index, bounce and dimension through a PCG hash (`shaders/athenea/material/sampling.slang`, "PCG (O'Neill)"), so 256 paths in one pass and 4 × 64 agree to relMSE $3.9\cdot10^{-15}$; between passes the camera is shifted within the pixel with Halton (2, 3) (§2.11, `modules/usd/src/Engine.cpp`), and the player tried the same jitter in its temporal antialiasing (§5.12). URL pcg-random.org; DOI (Halton) 10.1007/BF01386213.
:::

# Appendix A. Repositories and commands

## A.1 Code and demos

| repository | where | what it holds |
|---|---|---|
| athenea | github.com/jesusluque/athenea | the engine and renderer of §2: modules, shaders, the `hdAthenea` delegate, the `athenea` CLI, the Blender add-on `athenea_hydra` (`integrations/blender`, §2.14), the OpenFX node (`integrations/ofx`), the manuals `docs/operations.md` and `docs/development.md` with their Spanish translations, and `docs/decisions.md` |
| sparkwebGPU | sparkwebgpu.lucab.co.uk | the WebGPU player of §7, a fork of Spark: public demos of the `.athc` v3 format in the browser |

::: {.nota}
**The code and the demos, in detail** — What the public repository contains and where the player can be seen.

- **athenea** (github.com/jesusluque/athenea). The engine of §2 with its `athenea` CLI, its modules (`modules/<name>`, static libraries `athenea::<name>`), its shaders (`shaders/athenea/`), the AOFX bundles (`plugins/`: `mesh2splat`, `measure`, `splatbakefilter`, `splattransferzonal`, `invert`) and the manuals (`docs/operations.md`, `docs/development.md`, with their `.es.md` translations, and `docs/decisions.md`). The same repository holds the two integrations: the Blender add-on `athenea_hydra` (`integrations/blender/athenea_hydra`, §2.14) and the OpenFX node (`integrations/ofx`, built as `Athenea.ofx.bundle`). The code paths in these notes are relative to this repository.
- **sparkwebGPU** (public demos at https://sparkwebgpu.lucab.co.uk). The WebGPU player of §7, a fork of Spark: its Part II tools (`usd-athc`, `athc-convert`, `fit-compact-basis`, `floor_shadow`) derive the `.athc` v3 format from the master, its relight, compact-route, joint-CPCA-basis and skinning kernels are athenea kernels recompiled to WGSL, and the player draws them in the browser.
:::

## A.2 Commands and options

The manual for every subcommand and option is `docs/operations.md` (Spanish: `docs/operations.es.md`), which gives each
option's default, unit and constraint. The commands this paper relies on:

| command | used for | manual section | § |
|---|---|---|---|
| `athenea mesh2splat STAGE` (`--cell-from-camera`, `--camera-pixels`, `--transfer`, `--hide`, `--max-cells`, `--specular-filter`, `--edge-reach`, `--shadow-catcher`, `--light-catcher`, `--validate`, `--skinned`, `--lod-levels`) | conversion and bake | 2.8, 3.1 | 3, 4, 5 |
| `athenea decimate` (`--tree`, `--passes`, `--merge-glass`, `--object-groups`, `--tree-coverage`, `--tree-pages`, `--occlusion-mask`, `--occlusion-group`, `--cut-budget`, `--cut-cull-back`) | merge trees, cuts, pages, occlusion | 2.4.1, 3.3 | 5.9, 6 |
| `athenea visibility` | per-part visibility fields | 2.9 | 5.8 |
| `athenea stage`, `athenea view` | timings (`--frames`) and the viewer | 2.5, 2.6 | 2 |
| `athenea compare` | GPU image comparison | 2.11 | 2.15 |
| `athenea aofx` | the AOFX effects (mesh2splat, Measure) | 2.10, 7 | 2.14, 3.1 |
| `athc-convert` (`--gzip --planes`, `--shadow-vq`), `fit-compact-basis`, `floor_shadow --merge 2`, `usd-athc` | the player's formats (sparkwebGPU) | — | 7 |
| `athenea-mcp`, tool `render` (`xform`, `raw`, `layers`, `sky`, `skyRotation`, `skyVisible`, `splatView`) | a frame for a compositor, the sky swapped, the parameter channels | 6 | 2.14, 4.3, 4.11 |
| `Athenea.ofx.bundle` (`ATHENEA_BUILD_OFX`) | the OpenFX node in Nuke and Natron | 6.1 | 2.14 |

::: {.nota}
**How to read the A.2 notes** — For each command in the table, a note with each listed option: what it does, its default, its unit and the code that implements it. Values are taken from `docs/operations.md` (§2.4.1–§2.11, §6, §6.1) and from the CLI11 declarations (`add_option`/`add_flag`) in `apps/athenea/src/`; those of the player's tools, from sparkwebGPU's tools themselves.
:::

::: {.nota}
**`athenea mesh2splat STAGE`** — The conversion of §3 and the bake of §4–§5. Options are declared in `apps/athenea/src/CmdMesh2Splat.cpp` (class `Converter`); the emission kernel is the AOFX bundle `rt.sparrow.aofx.mesh2splat` (`plugins/mesh2splat/mesh2splat.slang`). The output is written atomically (a `.partial-<pid>` file renamed when done).

- `--cell-from-camera CAM` — path of a camera prim; no default. Each piece's cell is what one pixel of that camera covers where the mesh's box comes closest to it (at least the near plane), at `--time` (§3.3). It replaces `--density`; `--cell-min`/`--cell-max` only bound it if given. Code: `shaders/athenea/usd/mesh2splat_cells.slang`.
- `--camera-pixels P` — integer from 1 to 65 536, default 1920; pixels across the camera's horizontal aperture. The Corvette bakes used 640, 1280 and up to 5120 (§3.11).
- `--transfer` — flag, off. Bakes a TX transfer (how much sky reaches each Gaussian) instead of the light that arrived (§4): 112 values at degree 3 and 256 bits of open directions by default (`--transfer-degree` 3, `--transfer-cells` 16). Turns on `--specular-filter 1`. Code: the `kTransfer` variant of the path tracer (`modules/technique/src/PathTracer.cpp`, `modules/technique/include/athenea/technique/PathTracer.h`) through `StageRenderer::bakePointsOnDevice`, in slices (`Converter::transfer`); the field, in `shaders/athenea/usd/transfer_field_fill.slang` (`m2sFieldOverClosed`).
- `--hide P` — prim path, repeatable; none by default. Writes invisibility on `P` and everything under it in a session layer (the file does not change), before reading. It affects only the conversion: the bake traces the whole stage, so a part converted alone keeps the shadows and interreflections of the others (§3.2, §3.11, §6.9).
- `--max-cells` — integer, default 262 144 ($2^{18}$); the most cells a triangle walks over its projected box. The excess is counted and reported; at 4736 px `Infinite_Black` lost 359 628 and was redone with 1 048 576 (§3.3). Code: `plugins/mesh2splat/mesh2splat.slang`, `Mesh2Splat.cpp`.
- `--specular-filter k` — 0 to 4, unitless; default 1 with `--transfer` and 0 otherwise. Factor of the Toksvig widening: $s = k\max_e\angle/\lVert\Delta p\rVert\cdot\max(s_u,s_v)$, $\alpha'^2=\alpha^2+2s^2$ (§3.7). Variant B of §3.11 (0) shows the dark line at the skirt's fold. Code: `m2sSpread`, `m2sWidened` in `plugins/mesh2splat/mesh2splat.slang`.
- `--edge-reach ρ` — 0 to 8, in the Gaussian's own sizes; default 1.5; 0 sets no limit. The size across a hard edge (open, non-manifold or a fold of more than 40°, `kCreaseDegrees`) is limited to $\max(d/\rho, 0.05r)$ (§3.5). Code: `m2sHoldEdges` in `plugins/mesh2splat/mesh2splat.slang`; edges in `shaders/athenea/usd/mesh2splat_edges.slang`.
- `--shadow-catcher` — flag, off; needs `--prim OBJECT`. Turns the shadow the object casts on its ground (found, or `--catcher-ground`) into a patch of black Gaussians widened by `--catcher-margin` heights (1.5) with cell `--catcher-cell` (0 = one hundredth of the height), with a TX transfer bake; implies `--transfer` (§3.9). Code: `modules/usd/src/ShadowCatcher.cpp`.
- `--light-catcher` — flag, off; incompatible with `--transfer` and `--shadow-catcher`. The same patch, stretched by `--light-reach` object lengths (3), with a radiance bake per light group with the dome in black and drawn additively (§3.9). Writes `primvars:athenea:splat:lightCatcher`.
- `--validate DIR` — directory; no default. Measures the conversion material by material against the stage's path trace (`--validate-paths` 512, `--validate-bounces` 6, `--validate-size` 960 × 540, `--validate-sky` the scene's lights or `white` or an image), with Cryptomatte masks and the Measure effect, in `validate.json` (§3.10). Code: `apps/athenea/src/Mesh2SplatValidate.cpp`, `plugins/measure/measure.slang`.
- `--skinned` — flag, off. Reads the bind pose and keeps the skeleton: four joints, their weights and the gradients per Gaussian; forces `--no-bake`, stores a `--transfer` as two zonal lobes traced in the bind pose (`StageRenderer::holdBindPose`), and writes `skinningXforms` over `--range` (§5.2–§5.7). Code: `blendInfluences` in `plugins/mesh2splat/mesh2splat.slang`, the zonal fit in `plugins/splattransferzonal/`. It does not support `.athc` output.
- `--lod-levels N` — integer, default 1 (none). Converts again at half resolution for each level, each a `<name>_lod<n>.usdc` with its own rig, and `-o` becomes the scene that draws them as one cloud (§5.9). The engine picks the level with `Engine::lodLevelsFor` (threshold `athenea:lod:threshold`, 1 px) and does not pose the levels it does not draw (`deferredPose`; `modules/usd/src/Engine.cpp`).
:::

::: {.nota}
**`athenea decimate`** — Merge trees, cuts, pages and occlusion (§6). Options in `apps/athenea/src/CmdDecimate.cpp`; the work in `usd::decimateStage` (`modules/usd/src/Decimate.cpp`), `modules/usd/src/Occlusion.cpp` and the kernels `shaders/athenea/lod/lod_pairs.slang`, `lod_pairs_final.slang`, `lod_tree_cut.slang`, `lod_pages.slang`, `occlusion_mask.slang`. With `--ratio`, merging is pairwise (NanoGS); without it, the octree of §6.1 is cut.

- `--tree PATH` — with `--ratio`: also writes the whole merge tree, the leaves in depth order and then each merge, with `lodParent`, `lodStart`, `lodSize`, `lodCost` and the constant `lodTree` (§6.2.4). No default.
- `--passes N` — with `--ratio`: at most N pairwise passes; default 60. Each pass merges at most half of what is left, so a group of many small pieces needs more: 60 left 116 549 roots out of 323 613 leaves, and 1000 reached 1 298 in 310 passes; the whole Corvette used 4000 (§6.2.5).
- `--merge-glass` — flag, off; with `--ratio`. Transmitting Gaussians are also merged with one another; when off, glass is kept as is (roots equal leaves). The glass-with-non-glass guard always holds (§6.2.3, §6.2.5).
- `--object-groups K` and `--object-group i` — integers; off by default. Only the group of pieces whose `cryptoObject` modulo K is i is processed, for an asset that does not fit in one pass; since merges do not cross pieces, each group's tree is the exact subtree of the whole asset (§6.2.5).
- `--tree-coverage` — flag, off; the input is a merge tree. Each merge's opacity becomes what its leaves actually cover, the least-squares fit $a^\star=\sum GC/\sum G^2$ over an 8 × 8 grid on $[-1.25,1.25]^2$ (§6.4), with a floor of $2/255$; the mass is kept in `coverageMass` and the spill in `lodSpill`. Code: `treeCoverageToFile` in `lod_tree_cut.slang`.
- `--tree-pages P` — number of Gaussians per page; off by default (the paper uses 3000). Writes `lodPage`, `lodPageInfo` and `lodPageBounds`, ordering the nodes by the band of the parent's error, first leaf and size (§6.8). Code: `shaders/athenea/lod/lod_pages.slang`.
- `--occlusion-mask N` — codebook entries; off. The input must be a tree with 16 × 16 shadow bits: writes each node's conservative mask as an index into a codebook of N 256-bit entries (8192 for the car at 640 px, 8182 for the close-up level), §6.7. Code: `shaders/athenea/lod/occlusion_mask.slang`, `modules/usd/src/Occlusion.cpp`.
- `--occlusion-group PATHS` — with `--occlusion-mask`: more trees that share the input's codebook, allotted by need; a tree whose output is its own path is written alongside and renamed over it (`occlusionMasksShared`, §6.9).
- `--cut-budget N` — Gaussians; off. With an input tree, keeps the cut that would be drawn from `--camera` with at most N nodes, bisecting $\log\tau$ over $[10^{-4},10^4]$ in 40 steps (§6.3).
- `--cut-cull-back` — flag, off; needs `--cut-bounds`. Leaves out nodes whose whole normal cone faces away from the camera, only in closed shells that are not glass ($\theta<\pi/2\wedge n\cdot v<-\sin\theta$, §6.6). Code: `treeCulledMark` in `lod_tree_cut.slang`.
:::

::: {.nota}
**`athenea visibility`** — The bake of the per-part visibility fields of §5.8, on an already converted skinned cloud. Options in `apps/athenea/src/CmdVisibility.cpp`; the partition (`partitionJoints`) and the read (`visPartFrames`) in `modules/technique/src/SplatVisibility.cpp`; the bake in `shaders/athenea/splat/splat_visibility_bake.slang`, with inline rays (hence not on CUDA). The table lists no options; those that set the figures of §5.8 are `--parts` (12 parts), `--min-joints` (6; a smaller subtree stays with its parent's part), `--grid` (24 probes per axis of a part's box), `--octave` (16 directions per side of the octahedral map) and `--cut` (0.001, the transmittance below which a bake ray stops). `--skeleton-stage` and `--skeleton-prim` (the joint hierarchy) are required; `--prim` defaults to `/World/Splats`, and without `-o` it edits the file in place. On the sparrow: 22 s and 81 MB of fields.
:::

::: {.nota}
**`athenea stage`, `athenea view` (`--frames`)** — `--frames N` is the number of frames repeated for measuring. In `athenea stage` (`apps/athenea/src/CmdStage.cpp`) it defaults to 1 and reports the first, the median and the fastest; in `athenea view` (`apps/athenea/src/CmdView.cpp`, the window in `modules/view/src/Viewer.cpp`) it defaults to 0, which runs until the window is closed, and with N it closes after N frames and prints their timings. The paper's tables quote the median of the frames after the first, in milliseconds and with the release preset (§2.8, §2.10, §2.11, §5.6). The other options that weigh on the measurements: `--technique` (`raster` or `rt`), `--visibility` (`automatic`, `raster`, `rays`, `bvh`), `--size` (stage 1920x1080; view 1600x900 in points), `--path-samples`/`--path-bounces`/`--path-total` (stage 1/1/1; view 1/4/64).
:::

::: {.nota}
**`athenea compare`** — The image comparison of §2.15 as a command: with one image it gives per-channel means and maxima; with two (`image` and `reference`, EXRs of the same size) it gives the relMSE $\overline{(a-b)^2/(b^2+10^{-2})}$, the p99 and the largest relative difference, the distribution in 8-bit sRGB codes (p99, maximum, pixels above 2), the PSNR of the encoded colour and the mean luminance SSIM over a 7 × 7 box. Options: `--window X0 Y0 X1 Y1` (pixels, rows from the bottom; affects only the means), `--heatmap PATH` (the effect's EXR), `--show` (`source`, `difference`, `relative`, `codes`; default `codes`), `--gain` (≥ 0, default 1, the image only), `--path`. Every number is computed by a kernel of the Measure effect `rt.sparrow.aofx.measure` (`plugins/measure/measure.slang`), run by the AOFX host (`apps/athenea/src/CmdCompare.cpp`).
:::

::: {.nota}
**`athenea aofx`** — Two subcommands (`apps/athenea/src/CmdAofx.cpp`) over aopenfx's shared host, of which the engine only declares what it brings (`modules/aofx/host/src/Host.cpp`, §2.14). `athenea aofx list` prints each bundle found, loaded or rejected, with its effects, inputs and parameters; `athenea aofx run EFFECT INPUTS…` runs an effect on EXRs, with `-o` (default `out.exr`), repeatable `--param name=value`, `--time` (frame time, 0) and `--path` (directories searched after `$AOFX_PLUGIN_PATH`). The paper's bundles are `rt.sparrow.aofx.mesh2splat` (§3.1), `rt.sparrow.aofx.measure` (§2.15, §3.10), `rt.sparrow.aofx.splatbakefilter` (the à-trous filter of §4.4.4) and `rt.sparrow.aofx.splattransferzonal` (§4.4.5).
:::

::: {.nota}
**The player's tools (sparkwebGPU)** — None belongs to the engine; they derive the web format from the master (Part II).

- `usd-athc` — reads the USDC master with the `openusd` crate and writes a sectioned `.athc` v3, transfer included (`--transfer full|112|84|…|none`); with `--tree` it reads the merge tree of `athenea decimate --tree` and keeps its nodes, and with `--lod-errors` it writes the `LODE` section ($\varepsilon$, $\rho$, cone). `--add` joins several clouds in one file and `--only-prim`/`--exclude-prim` filter by the `cryptoManifest` (§7, step 1; §5.12 for skinned clouds).
- `athc-convert --gzip --planes` — `--gzip` compresses each section; `--planes` picks per section the smallest encoding among 0 (words), 1 (byte planes) and 2 (delta planes) (§7.1: light Corvette −10.6 %, HD −13.9 %).
- `athc-convert --shadow-vq C[,R[,P]]` — the 256 bits of open directions as a dictionary: k-means in Hamming space with majority centroids, C centroids (1024), R rounds (12), chunks of P splats (16 384), and it prints the p50/p90/p99 Hamming distances (§7.3). The asset's shared 4096-entry dictionary is the default on the compact route.
- `fit-compact-basis` — fits the compact tier's joint CPCA basis over an asset's pieces: `--clusters` (256; 512 in v8), `--iters` (16 Lloyd rounds), `--seed` (1), `--seed-sample` (200 000), `--sample S` (one node in S, for the whole car), `--held-out` (pieces measured without fitting); writes `<out>.json` and `<out>.bin`, which `usd-athc --compact-basis` reads (§7.2.2).
- `floor_shadow --merge 2` — turns a shadow catcher's `.athc` into the contact shadow's `FSH1` file; `--merge N` merges groups of N × N neighbours by moments (centre and covariance weighted by opacity × area, mass preserved, cells by majority): 309 k Gaussians to 78 k (§7.9).
:::

::: {.nota}
**`athenea-mcp`, tool `render`** — The engine as an MCP server (JSON-RPC 2.0 over stdio; `apps/athenea-mcp/main.cpp`, tools in `modules/mcp/src/Tools.cpp`), with the device and the stage kept warm between calls (§2.14). The `render` arguments the paper uses:

- `xform` — sixteen numbers, the camera-to-world matrix row by row, looking down −z with +y up; with `focal`, `haperture`, `vaperture` in millimetres and `near`/`far` in scene units. It is the camera as a compositor stores it.
- `raw` — path of a file that receives the frame's numbers instead of an image: RGBA as four float32 per pixel and then the view depth (0 where nothing was drawn), bottom row first, linear and premultiplied; the reply is text only. It is what the OpenFX node asks for.
- `layers` — a list; after the depth it adds four floats per pixel of `P` (world position, which needs a camera of its own, `xform` or `eye`) and of `CryptoObject00`–`02` (id, coverage, id, coverage); a frame with a matte also returns the manifest.
- `sky` — a lat-long `.hdr` or `.exr` image in every dome (empty restores the stage's); `skyRotation` — degrees about the scene's vertical axis; `skyVisible` — boolean; at `false` the background comes out transparent while the sky still lights (§4.3: a rotated cloud reads its transfer in its own frame with the sky rotated into it).
- `splatView` — the parameter channel of §4.11 drawn instead of the light (`albedo`, `roughness`, `metallic`, `transmission`, `ior`, `normal`, `opacity`, `size`, `skyVisibility`, `bounce`, `openCells`, `emission`, `field`, `baked`, `materialRoughness`; `light` returns to the light), the `athenea:splatView` setting.
:::

::: {.nota}
**`Athenea.ofx.bundle` (`ATHENEA_BUILD_OFX`)** — `ATHENEA_BUILD_OFX` is a CMake option, on by default (`CMakeLists.txt`), that builds on macOS and Linux (not on Windows) `<build>/ofx/Athenea.ofx.bundle` from `integrations/ofx/src` (`AtheneaOfx.cpp`, `RenderServer.cpp`) over Natron's OpenFX. The node does not link the engine: it starts `athenea-mcp` as a child process (the one in the bundle's `Contents/Resources/bin`, or another named by the `ATHENEA_MCP` variable) and reads each frame from a temporary file with `raw`. Its parameters: `cloud` (the cloud), `translate` (0 0 5), `rotate` (degrees) with `rot_order` (ZXY, Nuke's), `focal` (50 mm), `haperture` (24.576 mm), `vaperture` (0 = fitted to the format), `near`/`far` (0.1/10000), `time_offset` (frames), `sky`, `sky_rotation` (degrees), `sky_visible` and `output` (`rgba`, `depth`, `P`, `CryptoObject00`–`02`, the layer a host without planes shows). It is installed in `/Library/OFX/Plugins` or in `OFX_PLUGIN_PATH` (§2.14, manual §6.1).
:::

Environment variables (`ATHENEA_GPU_BUDGET`, `ATHENEA_STAGES`) are listed in `docs/operations.md` §8.1, the formats
(`.athc`, the codeless schemas) in §8.2 and §4.3.

::: {.nota}
**Environment variables and formats** — `ATHENEA_GPU_BUDGET` is the device memory a run may occupy, in MiB; without it the budget is Metal's `recommendedMaxWorkingSetSize` (18 186 MiB on the M5 Pro) and on CUDA and Vulkan there is none. `Device::admit` rejects any allocation that would exceed it, and on Apple silicon an allocation must also fit in the free physical memory minus 1.5 GiB (§2.3; `modules/gpu/src/Device.cpp`). `ATHENEA_STAGES=1` prints one line per frame with the time of each stage of the splat rasteriser (commit, per-part visibility, projection, counts, depth sort, emission, tile sort, blend) and of the mesh layer, waiting on each one, so the frame is slower (§2.5.3; `modules/usd/src/Engine.cpp`). In the current manual, §8.1 points to §1.2 of `docs/operations.md` (for `ATHENEA_GPU_BUDGET`) and to §3 of `docs/development.md` (for `ATHENEA_STAGES`); the codeless schemas (`AtheneaSplatLightingAPI`, `AtheneaStreamedAssetAPI` and others) are in §4.3 and the formats in §8.2.
:::

# Appendix B. Glossary {.unnumbered}

Technical terms of the paper, in alphabetical order, with the section where they appear.

à-trous filter
: Splat bake filter that averages neighbours with gaps that double at each iteration, weighted by tangent-plane position, orientation and prim id. It filters the indirect half and the reflected field, never the direct half (§4.4.4, `--bake-filter`, 3 iterations).

Adaptive sampling
: The path tracer's adaptive sampling: a pixel stops when the estimated relative error of its mean falls below $\varepsilon$, using the larger of its own variance and the mean of its 3 × 3 neighbours' (§2.11).

Affine approximation (EWA)
: Approximation of the perspective projection by its Jacobian at the Gaussian's centre, which turns a 3D Gaussian into a 2D one. It is exact for orthographic views and is the source of the difference between the rasteriser and the ray tracer (§2.5.1, §2.15).

AgX
: Troy Sobotka's view transform, used in Wrensch's analytic fit: log2 over $[-12.47, 4.03]$ stops and a sixth-order sigmoid (§2.12).

Albedo
: The fraction of light a diffuse surface reflects, the base colour without lighting; $\rho$ in the notation of §4.1 and the `albedo` channel of §4.11.

Alpha compositing
: Front-to-back compositing $C=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$, the over operator; the rasteriser, the reference and the ray tracer match its analytic form at p99 0 (§2.4, §2.5.2).

Ambient occlusion
: The fraction of the hemisphere from which sky arrives; in athenea it comes from the constant term of the transfer, $o=\operatorname{sat}(T_0/0.28209)$, and is the `skyVisibility` channel (§4.5, §4.11).

Anisotropic Gaussian
: A 3D Gaussian with three different scales along rotated axes, $\Sigma=RSS^\top R^\top$; the primitive of the whole system (§1.1, §2.4).

Antialiasing filter
: Dilation of the screen-space covariance by 0.3 px², with the opacity scaled by $k=\sqrt{\det\Sigma/\det\Sigma'}$ to preserve energy (§2.5.1, §7.7).

Any-hit ray
: A ray that accepts any intersection rather than the closest; the open-directions bake traces one per cell to know whether the direction leaves the scene (§4.4.2).

AOFX
: aopenfx's effects API (the `sdk/` SDK and the `host/` host), with an ABI that changes only additively; mesh2splat, Measure and the bake filter are AOFX bundles (§2.14, §3.1).

AOV
: Arbitrary output variable: a render output other than colour (depth, ids, normal, Cryptomatte), converted on the device only when a host maps the buffer (§2.13).

Argument buffer
: Metal's mechanism for passing many resources as a table; the `TextureStore`'s 1024-texture `ParameterBlock` is bound this way (§2.9).

Back-face culling
: Discarding nodes whose whole normal cone faces backwards, only in closed shells that are not glass; it speeds up the door by 12 % and removes the crack network between the two skins (§6.6, §7.5.1).

Bake
: Computing once, ahead of time, something that drawing only reads: here, light transport at each Gaussian with the path tracer (§4) or the per-part visibility fields (§5.8).

Band (error band)
: Octave of the error of a node's parent, $b(n)=\operatorname{clamp}(\lfloor\log_2\varepsilon(\mathrm{par}(n))\rfloor+40,0,63)$; pages are ordered by band (§6.8, §7.5.1).

Barycentric coordinates
: Weights $(\lambda_0,\lambda_1,\lambda_2)$ that express a point of a triangle as a combination of its vertices; they place each emitted Gaussian and blend the joint influences (§3.4, §5.3).

Bind pose
: The pose in which the geometry was bound to the skeleton; the skinned cloud is stored in it, and the transfer and visibility bakes are done in it (§5.1, §5.7).

BLAS / TLAS
: Bottom- and top-level acceleration structures of hardware ray tracing: one per mesh, and one per frame over the instances (§2.8).

Blend (stage)
: Last stage of the tile rasteriser: a group of 16 × 16 threads per tile walks its pairs front to back and composites each pixel (§2.5.2).

Body
: The part of the relight that does not depend on the eye, $\mathbf B=\sum_k L_k(T_k^{\text{dir}}+T_k^{\text{ind}})$ plus the sun and its bounce; it is cached while neither the sky nor the lights change (§4.5).

Bounce
: A bounce of a path after the first hit; also the channel that shows the indirect half under a white sky (§2.11, §4.11).

Bounding sphere
: Sphere $\rho(n)$ of each node that contains those of its children; together with the monotonic error it makes the cut a single parallel test (§6.3).

BSDF
: Bidirectional scattering distribution function: how a surface scatters light between two directions; in athenea a MaterialX BSDF node pushes lobes onto a `LobeStack` (§2.9).

BVH
: Bounding volume hierarchy: a tree of boxes for tracing rays; athenea uses the hardware one and a compute LBVH (§2.6, §2.8).

Byte planes
: Lossless encoding that transposes an array into planes (byte 0 of every element, then byte 1…) so that gzip finds runs; encoding 1 of `.athc` v3 (§7.1).

Car crop
: The 900 × 420 crop of the car in the 1920 × 1080 frame over which the car's PSNR is measured (§1.6).

Cell
: The world-space length $h$ that a converted Gaussian represents; it comes from the model, from the mesh or from the footprint of one pixel of a camera (§3.3). Also, a direction of the octahedral grid of open directions (§4.4.2).

Chi-square test (χ²)
: Statistical test that compares histograms of sampled directions with the pdf integrated per bin; it validates lobes and lights (§2.9, §2.10).

Chunk
: A block of 65 536 consecutive splats in Morton order in the original `.athc` format, the streaming unit of the LOD of §6.1.

Clear coat
: A thin dielectric layer over the base (lacquer); in the Corvette's paint it is what the eye sees, with weight 1, roughness 0 and index 1.45 (§3.6, §4.6.3).

Clip
: A skeleton animation; on the sparrow each clip is a layer that only changes `skinningXforms`, offered as a variant (§5.5).

Close-up level
: The Corvette bake at 5120 px for close views: 24 merge trees, 74.5 M leaves, 87.4 GB (§3.11, §6.9).

Closed shell
: A mesh that is closed by the rule $n_{\text{open}}<2\sqrt{1.5T}$; its Gaussians carry `closed = 1` and admit back-face culling (§3.5, §6.6).

Clustered PCA (CPCA)
: Compression that clusters vectors and applies PCA within each cluster; it compresses the transfer in the file and on the phone's GPU (§7.2).

Codebook
: A dictionary of entries whose index replaces a datum; the occlusion one is conservative (§6.7) and the open-directions one is by majority (§7.3).

Command buffer
: A list of work submitted to the GPU; an out-of-memory failure in one is now returned as an error on the next `submit()` (§2.3).

Compact record
: The player's 40-byte per-Gaussian record in byte planes: position, albedo, opacity, scale, rotation, normal, curvature, material, cluster, weights and shadow (§7.4).

Compute kernel
: A general-purpose GPU program launched over threads; all arithmetic on scene data is one (§2.1, §2.3).

Conic
: The inverse of the screen-space covariance, $Q=\Sigma'^{-1}$, which evaluates the Gaussian at each pixel (§2.5.2).

Contact shadow
: The player's pass that splats the shadow catcher's black Gaussians over the dome before the car (§7.9).

Counting sort
: Sorting by counting into buckets; the ray tracer sorts its 256 entries into 128 buckets (§2.6) and the player's shade pass groups by cluster (§7.8).

Covariance
: The matrix $\Sigma$ that gives a Gaussian its shape and orientation; on screen, $\Sigma_{2D}$ (§2.5.1).

Coverage
: The fraction of what lies behind that a surface covers; the conversion treats all opacity as coverage and translates it into $\alpha$ per Gaussian (§3.4), and a merge receives its own by least squares (§6.4).

Crate
: USD's binary format (`.usdc`); its compressed arrays arrive in new buffers at every read, hence `_held` (§5.6).

Cryptomatte
: An ID-matte standard: a hash of the name, (id, coverage) pairs per pixel and a manifest; athenea accumulates six ranks and stores `cryptoObject` per Gaussian (§2.5.2, §2.14).

Curvature
: The three values $(S_{uu},S_{uv},S_{vv})$ of the shape operator stored per Gaussian (§4.2, §4.8).

Cut
: The set of nodes of a merge tree drawn for a view (§1.6, §6.3).

Decimation
: Reducing the number of Gaussians by merging them; the sparrow's keeps the rig (§5.9) and that of §6 builds merge trees.

Delta light
: A light or lobe whose density is concentrated in one direction (a point light, a perfect mirror), which cannot be reached by sampling from the other side and keeps weight 1 in the MIS (§2.11).

Denoiser
: A filter that removes the noise of a path trace; athenea uses OIDN's `RT` filter guided by albedo and normal (§2.11).

Depth key
: The 24-bit sort key `bit_cast<uint>(z) >> 7`, which preserves the order of positive floats (§2.5.1).

Device
: The GPU as slang-rhi sees it; "on the device" means on the GPU (§2.1).

Dielectric
: A non-conducting material (glass, lacquer) whose Fresnel depends on the index of refraction (§2.9, §4.6.4).

Direct half
: The part of the transfer from paths that escape right after the first vertex: scalar, geometry only, computed by quadrature in TX (§4.4.1, §4.4.2).

Dispatch
: The launch of a compute kernel; athenea dispatches in threads, rounding the groups up (§2.3).

Dome light
: A UsdLux environment light with a lat-long image; it is sampled with cosine weighting or by warping its mip chain (§2.10, §4.3).

Dual quaternion skinning
: Skinning that blends dual quaternions instead of matrices; not implemented for clouds (§5.5, §5.13).

Edge reach
: How far a Gaussian may extend beyond a hard edge, in its own sizes; 1.5 by default (§3.5).

Eigen decomposition (Jacobi)
: Diagonalisation of a merged covariance by Jacobi sweeps in the shader, with the thinnest axis placed third (§6.1, §6.2.1).

Elastic term
: The second term of the skinning Jacobian, $\sum_j(X_jq)\nabla w_j^\top$, which stretches a Gaussian where the weights vary (§5.4.2).

Emissive table
: A table of emissive triangles weighted by area and luminance that the path tracer samples as lights (§2.11).

Environment albedo
: The integral of the specular BRDF over the hemisphere (DFG, the split-sum term), $P$ in the polish; athenea uses MaterialX's `ggxDirectionalAlbedo` (§4.6.4).

EWA splatting
: Elliptical weighted average: projection of a Gaussian through the local affine approximation of the perspective (§1.5, §2.5.1).

EXR
: OpenEXR, a floating-point image format with arbitrary channels and layers; the engine's linear premultiplied output (§2.14).

Eye-dome lighting
: Screen-space shading of point clouds from depth (§2.7).

Filtered importance sampling
: Reading each sample at the mip level of its solid angle; it builds the prefiltered map (§4.3).

Fold
: An edge where two triangles turn by more than 40°; a hard edge (§3.5), and where the per-pixel turning is switched off and the lobe widens (§4.9.4).

Footprint
: What a Gaussian covers on screen or a pixel covers in the world; it sets the cell size (§3.3) and the number of tiles (§2.5.1).

Fresnel
: The fraction of light reflected at an interface as a function of angle; athenea uses the exact one for the coat instead of Schlick (§4.6.4).

Frustum
: The camera's visible truncated pyramid; a splat's centre is clamped to the frustum widened to $[-0.15,1.15]$ (§2.5.1).

GeomSubset
: A subset of a USD mesh's faces; those of the `materialBind` family split a mesh into pieces (§3.2).

GGX
: The Trowbridge–Reitz microfacet distribution with parameter $\alpha=r^2$; that of all specular lobes (§4.1).

gpe
: A GPU image library that adopts slang-rhi's device; it lets meshes be passed to the AOFX effect without copies (§2.2, §3.1).

Guard (merge guard)
: A rule that makes the cost of merging two nodes infinite: piece, glass, mirror, fold, gloss, colour edge (§6.2.3).

Hamming distance
: The number of differing bits between two masks; the metric of the open-directions k-means and of the occlusion codebook (§6.7, §7.3).

Hard edge
: An open, non-manifold edge, or a fold of more than 40°; it limits the reach of the neighbouring Gaussians (§3.5).

Harder kernel
: Spark's kernel for opacity $D>1$: $1-(1-g)^A$, $A=\exp((D^2-1)/e)$, the compositing of $A$ coincident Gaussians (§2.5.2).

HDRI
: A high-dynamic-range environment image used as the sky; the Corvette's is `autoshop_01_4k` (§1.6, §4.10.4).

Hydra
: OpenUSD's rendering architecture: scene indices, render delegates and their prims (§2.13).

IES profile
: An IESNA LM-63 photometric profile that modulates a light's radiance, never its density (§2.10).

Importance sampling
: Sampling directions with a density similar to the integrand to reduce variance (§2.9, §2.10).

Indirect half
: The part of the transfer from paths that escape after bouncing: rgb, the colour of what they met (§4.4.1).

Inline ray tracing
: Ray queries (`RayQuery`) inside a compute kernel, with no ray tracing pipeline; the Metal route (§2.3, §2.6).

Instancer
: A prim that draws copies of a prototype; `PointInstancer` in USD, composed on the device as Storm does (§2.8, §3.2).

Integrator
: Code that accumulates the contribution of the particles along a ray; the ray tracer's two routes share `rt_integrate.slang` (§2.6).

Irradiance
: Incident power per unit area; that of a body without transfer comes from nine convolved SH coefficients (§4.3).

Jacobian
: The matrix of derivatives of a map: of the perspective (§2.5.1), of a triangle's inverse projection (§3.4) or of the skinning map with the weight gradients (§5.4.2).

Jitter
: A random offset within a stratum or a pixel; Halton (2, 3) between path tracer passes (§2.11).

k-means
: Clustering that alternates assigning to the nearest centroid and recomputing the centroids; it fits the CPCA basis and the shadow dictionary (§7.2, §7.3).

KL divergence
: The Kullback–Leibler divergence between distributions; Runnalls's bound on its increase by a merge is the shape cost (§6.2.2).

Lattice
: The grid of cells over which the Gaussians are emitted; their footprints sum to $2\pi\sigma^2$ (§3.3, §3.4).

LBVH
: Karras's linear BVH built in parallel over Morton codes; the ComputeBvh route and bvh visibility (§2.6, §2.8).

Leaf
: A Gaussian with its bake, the leaf of a merge tree (§1.6).

Light BVH
: A BVH over the lights with a box, an orientation cone and a power per node, used to pick one light per sample (§2.10).

Light catcher
: A patch of Gaussians on the ground with a bake of the light the lamps cast, drawn additively (§3.9).

Linear blend skinning
: Skinning that blends the joints' transforms by weights, $\mu'=M\sum_j w_jX_j(G\mu)$ (§5.4.1).

Lobe
: One term of a BSDF with `eval`, `sample` and `pdf` (diffuse, dielectric, conductor, sheen…); also a zonal lobe of the transfer (§2.9, §4.4.5).

LobeStack
: A material's lobes, sampled with one-sample MIS by their albedos (§2.9).

LOD
: Level of detail: drawing fewer nodes where the view resolves no more; here, merge trees with a cut by error (§6).

Mahalanobis radius
: Distance normalised by the covariance; a splat is cut at $r^2_{\max}=\min(2\ln(255\alpha),9)$ (§2.5.1).

Master
: The bake's output USDC file, at full precision and judged against the path trace, from which the formats derive (Part I).

MaterialX
: A material-graph standard; athenea compiles it to Slang (§2.9).

MCP
: Model Context Protocol; `athenea-mcp` serves the engine over JSON-RPC 2.0 on stdio (§2.14).

Merge
: An internal node of a merge tree, the merging of two nodes by moments (§1.6, §6.2).

Merge tree
: A binary tree of pairwise merges per piece whose cut by error gives the LOD (§6.2).

mesh2splat
: EA's algorithm that turns a mesh into Gaussians, one per covered texel, and the athenea command and bundle that extend it (§3).

Microfacet model
: A model of a rough surface as tiny mirrors with a distribution of normals (GGX), shadowing and Fresnel (§2.9).

Mip chain
: A texture's chain of levels, each half the previous one; generated by a kernel that preserves the mean (§2.3).

MIS
: Multiple importance sampling: combining NEE and material sampling with the power heuristic (§2.11).

Moments
: The weighted sums $W$, $W\mu$, $W\Sigma$; LOD nodes are moment vectors that add up (§6.1, §6.2.1).

Morton code
: A code that interleaves the bits of the coordinates; it orders splats and defines octree cells and merge candidates (§6.1, §6.2.4).

Motion blur
: Blur from motion during the shutter; in the rasteriser, a rank-one update of the covariance (§2.5.1, §5.6).

NEE
: Next-event estimation: sampling a light directly at each vertex of the path (§2.11).

Occlusion mask
: A conservative per-node mask of the directions from which some leaf is seen; with the eye outside the asset, a node closed towards the eye is discarded (§6.7).

Octahedral mapping
: A parametrisation of the sphere over a square through an unfolded octahedron; used by normals, open directions, the prefiltered map and per-part visibility (§4.2, §4.3, §5.8).

Octree
: A tree that divides space into eight; the original LOD groups by Morton prefixes (§6.1).

OIDN
: Intel Open Image Denoise, the path tracer's denoiser (§2.11).

Opacity compensation
: Scaling the opacity to preserve energy when a splat is widened, and giving back the area lost to the cut (`paidOpacity`, §2.5.1).

Open directions
: The `shadowBits`: per cell of an octahedral grid, whether the bake ray left the scene; 256 bits in TX (§4.2, §4.4.2).

OpenColorIO
: A colour management library used as a shader compiler (§2.12).

OpenFX
: The plugin API of compositors (Nuke, Natron); the `Athenea 3DGS` node draws a cloud as a generator (§2.14).

OpenPBR
: The ASWF's layered uber-shader; the Corvette's paint is OpenPBR (§3.2, §3.6).

Oracle
: A reference implementation against which another is checked; athenea has no CPU oracle and compares with kernels and closed forms (§2.1, §2.15).

Page
: The loading unit of a merge tree, consecutive nodes in band order; P = 3000 (§6.8, §7.5.1).

ParticleField
: The USD schema for particle clouds; `ParticleField3DGaussianSplat` is the master's prim (§2.4, Part I).

Path tracer
: A Monte Carlo integrator that follows light paths with bounces; the reference truth and the bake's integrator (§2.11).

PCG
: O'Neill's family of random number generators; the path tracer's sampler folds its indices through a PCG hash (§2.11).

Piece
: The triangles of a mesh bound to one material, the unit of conversion and of merging (§3.2, §6.2).

Polish
: The base's specular reflection, distinct from the coat's (§4.6.3).

Pool
: The player's fixed-size memory in which pages and ranges reside, with eviction starting from the finest band (§7.6).

Power heuristic
: The MIS weight $w=p_l^2/(p_l^2+p_m^2)$ (§2.11).

Prefiltered map
: The sky without its sun convolved with GGX lobes in an eight-level octahedral chain (§4.3).

Prefix sum
: A parallel cumulative sum that gives write offsets; one of the engine's two primitives (§2.3).

Primvar
: A USD attribute interpolated over a prim; athenea's parameters are stored as `primvars:athenea:splat:*` (Part I).

PRT
: Precomputed radiance transfer: storing per point the SH projection of visibility times the cosine (§1.5, §4.4.1).

PSNR
: Peak signal-to-noise ratio in dB over display-encoded images (§1.6).

Radiance bake
: Storing per Gaussian the light that arrived under the stage's lighting; exact only for that lighting (§4).

Radix sort
: Sorting by digits; the block-wise one sorts 10 M pairs in 12.4 ms (§2.3).

Range request
: An HTTP request for a range of bytes; a (block, tier) pair of `.athc` v3 is one (§7.1, §7.6).

Refit
: Recomputing a BVH's boxes without changing its topology when a pose moves the particles (§5.6).

Reflected field
: 16 rgb SH coefficients per Gaussian of the radiance bounced per direction under a white sky, normalised over the closed directions (§4.4.3).

Relight
: Lighting an asset again under another sky and other lights at draw time, from its transfer (§1.1, §4).

relMSE
: Relative mean squared error over linear values, $\overline{(a-b)^2/(b^2+10^{-2})}$ in the engine (§1.6, §2.15).

Render delegate
: The Hydra backend that receives and draws the prims; `hdAthenea` (§2.13).

Residency
: What must be in memory to draw a cut; the pages hold about 4× the cut (§6.8).

RGB9E5
: A 32-bit colour format with three 9-bit mantissas and a shared 5-bit exponent; the player's emission and drawn colour (§2.4, §7.7).

Rig
: The skeleton with its weights and transforms, which the system carries instead of the frames (§5.1).

Roughness
: A material's perceptual roughness $r$, with $\alpha=r^2$ in GGX (§4.1).

Scene index
: A Hydra filter that transforms the scene before the delegate; `HdAtheneaSkelSplatSceneIndex` (§2.13, §5.5).

Schlick (generalized)
: Approximate Fresnel between a facing colour and a grazing one; the metal of OpenPBR and glTF with the F82 tint (§3.6, §4.6.4).

Shadow catcher
: A patch of black Gaussians on the ground whose opacity reproduces the object's shadow, $-\ln(\text{ratio})/2\pi$ (§3.9, §7.9).

Shape operator
: The symmetric matrix $S$ with $d\mathbf n=S\,d\mathbf x$, the curvature of the surface under a Gaussian (§3.7, §4.8).

Sharp lobe
: A smooth reflection evaluated per pixel in the blend, turning the mirror with the shape operator (§4.8.2).

Sheen
: The lobe for the sheen of fabrics and microfibres, in its Imageworks and Zeltner forms (§2.9, §4.6.4).

Shutter
: The interval during which the shutter is open; a dragged splat is convolved with a segment (§2.5.1).

Skeleton / joint
: UsdSkel's hierarchy of joints, and each joint; the sparrow has 609 joints (§5).

Skinning
: Deforming geometry by the joints' transforms; here, deforming each Gaussian with its full Jacobian (§5.4).

Slang
: The shading language of every kernel, compiled to Metal, Vulkan, CUDA and WGSL (§2.1).

slang-rhi
: Slang's rendering hardware layer, on which the engine runs (§2.1).

Spherical harmonics
: An orthonormal basis of functions on the sphere, index $k=l^2+l+m$; degree 2 has 9 coefficients and degree 3 has 16 (§4.1).

Spill
: The fraction of what a merge paints that its leaves do not paint; evaluated and not adopted (§6.5).

Splat
: A 3D Gaussian drawn by splatting; used as a synonym of Gaussian (§1.6).

Split sum
: Approximation of environment reflection as the product of the prefiltered environment and the BRDF's albedo (§4.6.3).

SSIM
: Structural similarity index, a measure of perceptual similarity (§3.5).

Stage
: A composed USD scene; the bake's input as production delivers it (Part I, §3.2).

Still frame
: The player's policy that, with the camera still for 300 ms, lowers $\tau$ to 0.5 px and draws a fine cut (§7.10).

Storm
: Hydra's reference rasteriser, used as the oracle for mesh visibility (§2.8).

Streaming
: Loading data in parts according to the view: chunks (§6.1), pages and ranges (§7.6).

Surfel
: A Gaussian with one scale exactly 0, a disc drawn by ray–plane intersection (§2.5.2, §3.4).

Thin lens
: A lens model with blur by the circle of confusion, $R=fA|1/z-1/z_f|$ (§2.5.1, §2.11).

Thin wall
: Sheet glass whose two faces do not deflect; it covers $2R/(1+R)$ and scales its reflection (§3.8, §4.7).

Throughput
: The product of a path's weights, $f|\cos\theta|/p$ per vertex (§2.11).

Tier
: A level of `.athc` v3 that groups sections so that one request brings what it needs (§7.1).

Tile
: A 16 × 16-pixel block of the rasteriser (§2.5).

Toksvig widening
: Widening the roughness by the spread of the normals under the Gaussian, $\alpha'^2=\alpha^2+2s^2$ (§3.7, §4.9).

Transfer
: The bake's light transport: how a Gaussian's surroundings would bring it any sky (§1.6, §4).

Transmittance
: The fraction of light that passes, $T=\prod(1-\alpha_j)$; the pixel stops below $10^{-4}$ (§2.5.2).

Traversal record
: A quantised 5–11-byte record that the player walks to choose the cut (§7.5.3).

TX
: The current form of the transfer: degree-3 direct and indirect SH, the reflected field and 256 bits of open directions (§1.6, §4.2).

UDIM
: A convention of texture tiles by UV ranges, resolved through an indirection table (§2.9).

Unified memory
: Memory shared by CPU and GPU on Apple silicon; several processes can exhaust it together (§2.3).

Unwidening
: Undoing the Toksvig widening when reading a sharp lobe, $r=\max(r'^4-2s^2,0)^{1/4}$ (§4.9.2).

UsdSkel
: USD's schema for skeletons and skinning (§5.2, §5.5).

Variant set
: A set of alternatives of a USD prim; the sparrow's clips are chosen this way (§5.5).

Vector quantization
: Replacing a vector by the index of its centroid; the open-directions bits (§7.3).

View transform
: The transform from linear scene values to the display: Standard, AgX or ACES 2.0 (§2.12).

Visibility buffer
: A buffer of (instance + 1, triangle) per pixel from which the hit is rebuilt (§2.8).

Visibility field (per part)
: A table of directional transmittance per body part, baked in the bind pose and read as a product (§5.8).

VNDF
: The distribution of visible normals, which the microfacet lobes sample (§2.9).

von Mises–Fisher
: A distribution of directions on the sphere; the reflected GGX is taken as one with concentration $1/2\alpha^2$ when the field is read (§4.6.2).

WASM
: WebAssembly; the player's traversal runs in a WASM worker (§7.5, §7.6).

WebGPU / WGSL
: The web's GPU API and its shading language, the basis of the player (§7).

White furnace
: A test with a uniform white environment in which a lossless material must return 1 (§2.9, §2.11).

Workgroup memory
: Memory shared by a group of threads; the blend loads its tile's records there (§2.5.2).

Zonal harmonics
: SH symmetric about an axis, which rotate by rotating the axis; two lobes per skinned Gaussian (§4.4.5, §5.7).
