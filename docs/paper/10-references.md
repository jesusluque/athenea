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

## 10.2 Meshes to Gaussians

12. Electronic Arts (SEED). mesh2splat: converting meshes into 3D Gaussian splats by rasterisation (open source,
    BSD-3). The origin of the conversion of §3.
13. Mesh2GS. arXiv 2606.21898 (2026). The camera-derived cell size of §3.3.

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

## 10.5 Geometry, acceleration structures and levels of detail

32. Karras, T. (2012). Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees. High Performance
    Graphics.
33. Möller, T., Trumbore, B. (1997). Fast, Minimum Storage Ray–Triangle Intersection. Journal of Graphics Tools.
34. Runnalls, A. R. (2007). Kullback–Leibler Approach to Gaussian Mixture Reduction. IEEE Transactions on Aerospace
    and Electronic Systems.
35. Karis, B., Stubbe, R., Wihlidal, G. (2021). Nanite: A Deep Dive. SIGGRAPH Advances in Real-Time Rendering.

## 10.6 Colour

36. Academy of Motion Picture Arts and Sciences (2024). ACES 2.0 output transforms (CTL reference).
37. Hellwig, L., Stolitzka, D., Fairchild, M. D. (2022). The colour appearance model known as Hellwig 2022, a revision
    of CAM16, used by the ACES 2.0 output transforms.
38. Wrensch, B. An analytic fit of the AgX view transform. Online note, as cited in §2.12.
39. Krystek, M. (1985). An Algorithm to Calculate Correlated Colour Temperature. Color Research and Application. (The
    blackbody fit of §2.10.)
40. OpenColorIO. Academy Software Foundation.

## 10.7 Animation and visibility

41. Lin, et al. Directional visibility MLPs per body part for articulated avatars. As cited in §5.8.
42. DNF-Avatar: a scalar ambient-occlusion grid per bone. As cited in §5.8.

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

# Appendix A. Repositories, commands and figures

## A.1 Repositories

| repository | where | what it holds |
|---|---|---|
| athenea | github.com/jesusluque/athenea | the engine and renderer of §2: modules, shaders, the `hdAthenea` delegate, the `athenea` CLI, the manuals `docs/operations.md` and `docs/development.md` with their Spanish translations, and `docs/decisions.md` |
| athenea-pieces | github.com/jesusluque/athenea-pieces, branch `pieces` (private) | a fork of athenea holding the pieces work: per-part merge trees, the error cut, closed shells, the shared occlusion, the 1280 and 5120 px bakes (§3.11, §6); `docs/pieces.md` and this paper under `docs/paper/` |
| sparkwebGPU | `~/luc/sparkwebGPU`, branch `webgpu` | the WebGPU fork of Spark of §7: the `.athc` v3 reader, the compact path, the streamer, the player pages; its `CLAUDE.md` log and `research/` notes |

Commit hashes quoted in Sections 3–6 are athenea-pieces commits; §2.15's 2e6e183 is a commit of lucabRTrender,
athenea's predecessor before it was renamed with a fresh history. Those in §7
are sparkwebGPU commits unless the text names athenea.

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

Environment variables (`ATHENEA_GPU_BUDGET`, `ATHENEA_STAGES`) are listed in `docs/operations.md` §8.1, the formats
(`.athc`, the codeless schemas) in §8.2 and §4.3. GPU jobs ran one at a time through `~/luc/athenea-gpu/shared.sh`,
under `guard.sh` and `memguard.sh` (`docs/pieces.md` §5).

## A.3 Figures

The figures referred to in Sections 2–7 are kept outside the paper's directory:

| figure | file | § |
|---|---|---|
| the sparrow, mesh path traced and converted | `athenea-pieces/docs/images/sparrow-mesh-cloud.png`, `sparrow-cloud.png`, `sparrow-film.png` | 2, 5.10 |
| edges held vs control (640 px car) | `athenea-renders/simplify3/edges/edges_cmp.png`, `day_new_Camera.png`, `day_ctl_Camera.png` | 3.5 |
| the 1280 px bake test A / B / C | `athenea-renders/pieces/car1280/test/A.png`, `B.png`, `C.png` | 3.11 |
| stored coat roughness, path trace vs leaves vs cut | `athenea-renders/pieces/car/t63b/research_coat_roughness.png`, `t63b/mos.png` | 4.10 |
| cut_a / cut_b; the door's open class | `athenea-renders/pieces/car/t65/mos.png`, `t66/research_open_split.png` | 4.10.4 |
| the 1280 px bake vs path trace; sill fold dashes | `athenea-renders/pieces/car/t71/mos.png`, `t71/research_sill.png` | 4.10, 6.10 |
| gated sharp polish; tube/fold split | `athenea-renders/pieces/car/t73/bump_mos.png`, `t74/sill_mos.png` | 4.10.6 |
| the sparrow on the web, path trace vs raster; the corrected wing | `athenea-renders/sparrow-web/ref_pt_raster.png`, `sparrow/fixed_wing_athenea.png` | 5.10 |
| the white pawn and its ring | `athenea-renders/pawn-hq/pawn_gt_gs.png`, `ring_gt_gs.png` | 5.11 |
| the crack net and back culling (T50) | `athenea-renders/pieces/car/t50/cut_Camera.png`, `cull_Camera.png` | 6.6, 6.10 |
| logo and grille at 1280 vs 5120 px | `athenea-renders/pieces/close/r_1280.png`, `r_5120.png`, `close_mos.png` | 3.11, 6.9, 7.12 |

All render paths are under `~/luc/athenea-renders/`; every render of the work is saved there by task.
