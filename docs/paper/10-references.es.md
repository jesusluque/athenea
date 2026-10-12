<!-- Copyright (c) 2026 jesus luque. -->

# 10. Referencias

Las referencias se dan por autores, título y año tal como se citan en el texto. Cuando las fuentes en las que se basa
este artículo nombran una obra sin sus datos bibliográficos completos, se enumera tal como se cita y se marca así.

## 10.1 Splatting de gaussianas y render basado en puntos

1. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field
   Rendering. ACM Transactions on Graphics (SIGGRAPH).
2. Kerbl, B., Meuleman, A., Kopanas, G., Wimmer, M., Lanvin, A., Drettakis, G. (2024). A Hierarchical 3D Gaussian
   Representation for Real-Time Rendering of Very Large Datasets. ACM Transactions on Graphics (SIGGRAPH).
3. Zwicker, M., Pfister, H., van Baar, J., Gross, M. (2001). EWA Volume Splatting / Surface Splatting. IEEE
   Visualization y SIGGRAPH; y (2002) EWA Splatting, IEEE Transactions on Visualization and Computer Graphics.
4. Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A. (2024). Mip-Splatting: Alias-free 3D Gaussian Splatting.
   CVPR.
5. Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S. (2024). 2D Gaussian Splatting for Geometrically Accurate Radiance
   Fields. SIGGRAPH.
6. Pfister, H., Zwicker, M., van Baar, J., Gross, M. (2000). Surfels: Surface Elements as Rendering Primitives.
   SIGGRAPH.
7. Moënne-Loccoz, N., et al. (2024). 3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes (3DGRT). ACM
   Transactions on Graphics (SIGGRAPH Asia).
8. Ren, K., et al. (2024). Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians.
9. Xiong, et al. (2026). NanoGS: fusión por pares de vecinos más cercanos mutuos de gaussianas. Tal como se cita en
   la §6.2.
10. GaussianPOP: una puntuación de poda por gaussiana según el error que deja su eliminación. Tal como se cita en la
    §2.5.2.
11. Spark: un renderizador de splatting de gaussianas 3D para la web (World Labs). El reproductor de la §7 es un fork
    con backend WebGPU.

## 10.2 De mallas a gaussianas

12. Electronic Arts (SEED). mesh2splat: conversión de mallas en splats de gaussianas 3D por rasterización (código
    abierto, BSD-3). El origen de la conversión de la §3.
13. Mesh2GS. arXiv 2606.21898 (2026). El tamaño de celda derivado de una cámara de la §3.3.

## 10.3 Transporte de luz, transfer precalculado y compresión

14. Sloan, P.-P., Kautz, J., Snyder, J. (2002). Precomputed Radiance Transfer for Real-Time Rendering in Dynamic,
    Low-Frequency Lighting Environments. SIGGRAPH.
15. Sloan, P.-P., Hall, J., Hart, J., Snyder, J. (2003). Clustered Principal Components for Precomputed Radiance
    Transfer. SIGGRAPH.
16. Sloan, P.-P. (2008). Stupid Spherical Harmonics (SH) Tricks. Game Developers Conference. (Ajuste de armónicos
    zonales.)
17. Ramamoorthi, R., Hanrahan, P. (2001). An Efficient Representation for Irradiance Environment Maps. SIGGRAPH.
18. Křivánek, J., Colbert, M. (2008). Real-time Shading with Filtered Importance Sampling. Computer Graphics Forum
    (EGSR).
19. Lagarde, S., de Rousiers, C. (2014). Moving Frostbite to Physically Based Rendering. Notas de curso de SIGGRAPH.
20. Toksvig, M. (2005). Mipmapping Normal Maps. Journal of Graphics Tools.
21. Veach, E., Guibas, L. (1995). Optimally Combining Sampling Techniques for Monte Carlo Rendering. SIGGRAPH. (La
    heurística de potencia de la §2.11.)
22. Conty Estevez, A., Kulla, C. (2018). Importance Sampling of Many Lights with Adaptive Tree Splitting. HPG. (La BVH
    de luces de la §2.10.)
23. Lloyd, S. (1982). Least Squares Quantization in PCM. IEEE Transactions on Information Theory; Arthur, D.,
    Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. SODA. (Los ajustes k-means de las
    §7.2–7.3.)

## 10.4 Materiales y sombreado

24. Walter, B., Marschner, S., Li, H., Torrance, K. (2007). Microfacet Models for Refraction through Rough Surfaces.
    EGSR.
25. Heitz, E. (2018). Sampling the GGX Distribution of Visible Normals. Journal of Computer Graphics Techniques.
26. Oren, M., Nayar, S. (1994). Generalization of Lambert's Reflectance Model. SIGGRAPH; Portsmouth, J., Kutz, P.,
    Hill, S. (2025). EON: A Practical Energy-Preserving Rough Diffuse BRDF.
27. Burley, B. (2012). Physically-Based Shading at Disney. Notas de curso de SIGGRAPH.
28. Estevez, A. C., Kulla, C. (2017). Production Friendly Microfacet Sheen BRDF. Sony Pictures Imageworks;
    Zeltner, T., Burley, B., Chiang, M. J.-Y. (2022). Practical Multiple-Scattering Sheen Using Linearly Transformed
    Cosines.
29. Lazarov, D. (2013). Getting More Physical in Call of Duty: Black Ops II. Notas de curso de SIGGRAPH. (El ajuste
    del albedo de entorno comparado en la §4.6.4.)
30. Andersson, Z., Kutz, P., et al. (2024). Especificación de OpenPBR Surface. Academy Software Foundation.
31. MaterialX: un estándar abierto para contenido de materiales y de look development (versión 1.39). Academy
    Software Foundation.

## 10.5 Geometría, estructuras de aceleración y niveles de detalle

32. Karras, T. (2012). Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees. High Performance
    Graphics.
33. Möller, T., Trumbore, B. (1997). Fast, Minimum Storage Ray–Triangle Intersection. Journal of Graphics Tools.
34. Runnalls, A. R. (2007). Kullback–Leibler Approach to Gaussian Mixture Reduction. IEEE Transactions on Aerospace
    and Electronic Systems.
35. Karis, B., Stubbe, R., Wihlidal, G. (2021). Nanite: A Deep Dive. SIGGRAPH Advances in Real-Time Rendering.

## 10.6 Color

36. Academy of Motion Picture Arts and Sciences (2024). Transformaciones de salida de ACES 2.0 (referencia en CTL).
37. Hellwig, L., Stolitzka, D., Fairchild, M. D. (2022). El modelo de apariencia de color conocido como Hellwig 2022,
    una revisión de CAM16, que usan las transformaciones de salida de ACES 2.0.
38. Wrensch, B. Un ajuste analítico de la transformación de vista AgX. Nota en línea, tal como se cita en la §2.12.
39. Krystek, M. (1985). An Algorithm to Calculate Correlated Colour Temperature. Color Research and Application. (El
    ajuste del cuerpo negro de la §2.10.)
40. OpenColorIO. Academy Software Foundation.

## 10.7 Animación y visibilidad

41. Lin, et al. MLP de visibilidad direccional por parte del cuerpo para avatares articulados. Tal como se cita en la
    §5.8.
42. DNF-Avatar: una rejilla escalar de oclusión ambiental por hueso. Tal como se cita en la §5.8.

## 10.8 Sistemas y estándares

43. Pixar Animation Studios. OpenUSD (Universal Scene Description) e Hydra, incluidos UsdSkel, UsdLux, UsdShade y el
    esquema `ParticleField`.
44. He, Y., Fatahalian, K., Foley, T. (2018). Slang: Language Mechanisms for Extensible Real-time Shading Systems. ACM
    Transactions on Graphics; slang-rhi, la interfaz de hardware de render que usa el motor.
45. Intel. Open Image Denoise (OIDN) 2.5.
46. W3C. WebGPU y el WebGPU Shading Language (WGSL).
47. Friedman, J., Jones, A. C. (2015). Fully Automatic ID Mattes with Support for Motion Blur and Transparency
    (Cryptomatte). SIGGRAPH Posters.
48. O'Neill, M. E. (2014). PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number
    Generation; Halton, J. H. (1960). On the efficiency of certain quasi-random sequences of points.

# Apéndice A. Repositorios, comandos y figuras

## A.1 Repositorios

| repositorio | dónde | qué contiene |
|---|---|---|
| athenea | github.com/jesusluque/athenea | el motor y renderizador de la §2: módulos, shaders, el delegado `hdAthenea`, la CLI `athenea`, los manuales `docs/operations.md` y `docs/development.md` con sus traducciones al español, y `docs/decisions.md` |
| athenea-pieces | github.com/jesusluque/athenea-pieces, rama `pieces` (privado) | un fork de athenea con el trabajo de pieces: árboles de fusión por pieza, el corte por error, las cáscaras cerradas, la oclusión compartida, los horneados de 1280 y 5120 px (§3.11, §6); `docs/pieces.md` y este artículo en `docs/paper/` |
| sparkwebGPU | `~/luc/sparkwebGPU`, rama `webgpu` | el fork WebGPU de Spark de la §7: el lector de `.athc` v3, el camino compacto, el streamer, las páginas del reproductor; su registro `CLAUDE.md` y sus notas `research/` |

Los hashes de commit citados en las secciones 3–6 son commits de athenea-pieces; el 2e6e183 de la §2.15 es un commit de
lucabRTrender, el predecesor de athenea antes de que se renombrara con una historia nueva. Los de la §7
son commits de sparkwebGPU salvo que el texto nombre athenea.

## A.2 Comandos y opciones

El manual de cada subcomando y opción es `docs/operations.md` (en español: `docs/operations.es.md`), que da el valor
por defecto, la unidad y la restricción de cada opción. Los comandos en los que se apoya este artículo:

| comando | se usa para | sección del manual | § |
|---|---|---|---|
| `athenea mesh2splat STAGE` (`--cell-from-camera`, `--camera-pixels`, `--transfer`, `--hide`, `--max-cells`, `--specular-filter`, `--edge-reach`, `--shadow-catcher`, `--light-catcher`, `--validate`, `--skinned`, `--lod-levels`) | conversión y horneado | 2.8, 3.1 | 3, 4, 5 |
| `athenea decimate` (`--tree`, `--passes`, `--merge-glass`, `--object-groups`, `--tree-coverage`, `--tree-pages`, `--occlusion-mask`, `--occlusion-group`, `--cut-budget`, `--cut-cull-back`) | árboles de fusión, cortes, páginas, oclusión | 2.4.1, 3.3 | 5.9, 6 |
| `athenea visibility` | campos de visibilidad por parte | 2.9 | 5.8 |
| `athenea stage`, `athenea view` | tiempos (`--frames`) y el visor | 2.5, 2.6 | 2 |
| `athenea compare` | comparación de imágenes en la GPU | 2.11 | 2.15 |
| `athenea aofx` | los efectos AOFX (mesh2splat, Measure) | 2.10, 7 | 2.14, 3.1 |
| `athc-convert` (`--gzip --planes`, `--shadow-vq`), `fit-compact-basis`, `floor_shadow --merge 2`, `usd-athc` | los formatos del reproductor (sparkwebGPU) | — | 7 |

Las variables de entorno (`ATHENEA_GPU_BUDGET`, `ATHENEA_STAGES`) se enumeran en la §8.1 de `docs/operations.md`, los
formatos (`.athc`, los esquemas sin código) en la §8.2 y la §4.3. Los trabajos de GPU se ejecutaron de uno en uno
mediante `~/luc/athenea-gpu/shared.sh`, bajo `guard.sh` y `memguard.sh` (§5 de `docs/pieces.md`).

## A.3 Figuras

Las figuras a las que se refieren las secciones 2–7 se guardan fuera del directorio del artículo:

| figura | fichero | § |
|---|---|---|
| el gorrión, malla trazada y convertida | `athenea-pieces/docs/images/sparrow-mesh-cloud.png`, `sparrow-cloud.png`, `sparrow-film.png` | 2, 5.10 |
| bordes sujetos frente al control (coche de 640 px) | `athenea-renders/simplify3/edges/edges_cmp.png`, `day_new_Camera.png`, `day_ctl_Camera.png` | 3.5 |
| la prueba de horneado A / B / C a 1280 px | `athenea-renders/pieces/car1280/test/A.png`, `B.png`, `C.png` | 3.11 |
| rugosidad del coat guardada, path trace frente a hojas frente a corte | `athenea-renders/pieces/car/t63b/research_coat_roughness.png`, `t63b/mos.png` | 4.10 |
| cut_a / cut_b; la clase abierta de la puerta | `athenea-renders/pieces/car/t65/mos.png`, `t66/research_open_split.png` | 4.10.4 |
| el horneado de 1280 px frente al path trace; los trazos del pliegue del umbral | `athenea-renders/pieces/car/t71/mos.png`, `t71/research_sill.png` | 4.10, 6.10 |
| polish nítido condicionado; separación tubo/pliegue | `athenea-renders/pieces/car/t73/bump_mos.png`, `t74/sill_mos.png` | 4.10.6 |
| el gorrión en la web, path trace frente a raster; el ala corregida | `athenea-renders/sparrow-web/ref_pt_raster.png`, `sparrow/fixed_wing_athenea.png` | 5.10 |
| el peón blanco y su anillo | `athenea-renders/pawn-hq/pawn_gt_gs.png`, `ring_gt_gs.png` | 5.11 |
| la red de grietas y el cull de caras traseras (T50) | `athenea-renders/pieces/car/t50/cut_Camera.png`, `cull_Camera.png` | 6.6, 6.10 |
| logo y rejilla a 1280 frente a 5120 px | `athenea-renders/pieces/close/r_1280.png`, `r_5120.png`, `close_mos.png` | 3.11, 6.9, 7.12 |

Todas las rutas de render están bajo `~/luc/athenea-renders/`; cada render del trabajo se guarda allí por tarea.
