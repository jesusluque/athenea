<!-- Copyright (c) 2026 jesus luque. -->

# 10. Referencias

Las referencias se dan por autores, título y año tal como se citan en el texto. Cuando las fuentes en las que se basa
este artículo nombran una obra sin sus datos bibliográficos completos, se enumera tal como se cita y se marca así.

## 10.1 Gaussian splatting y render basado en puntos

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
9. Xiong, et al. (2026). NanoGS: merge por pares de vecinos más cercanos mutuos de Gaussians. Tal como se cita en
   la §6.2.
10. GaussianPOP: una puntuación de poda por Gaussian según el error que deja su eliminación. Tal como se cita en la
    §2.5.2.
11. Spark: un renderer de Gaussian splatting 3D para la web (World Labs). El player de la §7 es un fork
    con backend WebGPU.

::: {.nota}
**Notas a 10.1. Gaussian splatting y render basado en puntos** — Qué aportó cada obra y dónde la usa o se aparta de ella athenea. Las rutas de código son relativas al repositorio público github.com/jesusluque/athenea.

**[1] Kerbl et al. 2023, 3D Gaussian Splatting (3DGS)** — Representa una escena como $N$ Gaussians 3D anisótropas, cada una con media $\mu$, covarianza $\Sigma = R S S^\top R^\top$ (cuaternión y escalas en logaritmo), opacidad $\alpha$ y color en spherical harmonics de grado 3; las optimiza por render diferenciable a partir de fotografías, con densificación adaptativa, y las dibuja con un rasterizer por tiles que ordena claves (tile, profundidad) y compone de delante hacia atrás, $C=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$. athenea toma la representación y la familia del rasterizer (§2.4–§2.5: `render::TileRasterizer`, `shaders/athenea/splat/splat_project.slang`, `splat_blend.slang`, el layout en `shaders/athenea/common/packing.slang`) y la acotación del centro al frustum ensanchado. Se aparta en que no entrena: las Gaussians salen de meshes (§3) y su luz de un bake (§4); parte la sort de 64 bits en una sort por profundidad de $N$ claves y una sort estable por tile (§2.5); corta en $r^2_{\max}(\alpha)$ en vez de en $3\sigma$; y mezcla en luz lineal. DOI 10.1145/3592433; arXiv 2308.04079.

**[2] Kerbl et al. 2024, Gaussians 3D jerárquicas** — Construye sobre una escena entrenada una jerarquía de nodos cuyas Gaussians fusionadas salen de momentos ponderados (peso por opacidad y área) y la corta por tamaño en pantalla, con interpolación entre niveles y entrenamiento por trozos para escenas muy grandes. El LOD original de athenea (§6.1) sigue esos momentos, $W=\sum w$, $S_1=\sum w\mu$, $S_2=\sum w(\Sigma+\mu\mu^\top)$ con $w=o\,A_2(s)$, pero los agrupa en un octree por código de Morton y no en una jerarquía propia (`shaders/athenea/lod/lod_common.slang`, que cita la obra; `modules/lod/src/Lod.cpp`, `LodBuilder`). Los merge trees de §6.2 conservan la suma de momentos pero fusionan por pares y toman el segundo momento respecto al centro del grupo. DOI 10.1145/3658160; arXiv 2406.12080.

**[3] Zwicker et al. 2001–2002, EWA splatting** — El filtro de media ponderada elíptica (elliptical weighted average) proyecta un núcleo de reconstrucción Gaussiano 3D con la aproximación afín local de la perspectiva (el Jacobian $J$ en el centro), lo que da una Gaussiana 2D $\Sigma' = J W \Sigma W^\top J^\top$, y la convoluciona con un filtro paso bajo de pantalla para eliminar el aliasing. La fórmula de §2.5.1, $\Sigma_{2D}=(JWM)(JWM)^\top$, es exactamente esta (`shaders/athenea/splat/frame.slang`, `splat_project.slang`; la referencia en `shaders/athenea/reference/reference_project.slang`). El ray tracer de §2.6 evita la aproximación evaluando el pico de cada partícula a lo largo del rayo, y §2.15 explica que el raster coincide con él a p99 2 solo donde EWA es exacto (ortográficas). DOI 10.1109/VISUAL.2001.964490 (EWA Volume Splatting), 10.1145/383259.383300 (Surface Splatting), 10.1109/TVCG.2002.1021576 (EWA Splatting).

**[4] Yu et al. 2024, Mip-Splatting** — Añade a 3DGS un filtro 3D que limita la frecuencia de cada Gaussian a la máxima frecuencia de muestreo de las vistas de entrenamiento, y sustituye la dilatación de pantalla por un filtro Mip 2D que conserva la energía escalando la opacidad por $\sqrt{\det\Sigma/\det(\Sigma+sI)}$. athenea pertenece a esa familia (§1.5, §2.5.1): dilata 0.3 px² y escala el pico por $k=\sqrt{\det\Sigma_{\text{still}}/\det\Sigma'}$, una sola raíz que cubre filtro, shutter y lente; añade además `paidOpacity`, que devuelve el área perdida por el corte en 1/255 (`shaders/athenea/splat/splat_project.slang`, `frame.slang`). No tiene filtro 3D: el tamaño de cada Gaussian lo fija la cell de la conversión (§3.3). arXiv 2311.16493.

**[5] Huang et al. 2024, 2D Gaussian Splatting** — Aplana cada Gaussian en un disco orientado (dos escalas tangentes) y calcula la intersección rayo–disco exacta como el cruce de los dos planos homogéneos del píxel, sin aproximación afín; un filtro paso bajo en espacio de objeto toma el máximo entre el disco y una Gaussian de pantalla para vistas de canto. En athenea una Gaussian con una escala exactamente 0 es un surfel que el rasterizer dibuja así, con el filtro como máximo frente a una Gaussian redonda de varianza ½ px² y la opacidad devuelta por área (§2.5.2; `shaders/athenea/splat/splat_project.slang`, `splat_blend.slang`, `modules/render/include/athenea/render/TileRasterizer.h`). La conversión los escribe con `--flatness 0` (§3.4) y los merges planos se escriben como surfels (`shaders/athenea/lod/lod_pairs_final.slang`, §6.2.1). No usa los regularizadores de profundidad ni de normales, que son de entrenamiento. DOI 10.1145/3641519.3657428; arXiv 2403.17888.

**[6] Pfister et al. 2000, Surfels** — Define el surfel (surface element) como primitiva de render: un punto con posición, normal, radio y atributos prefiltrados, dibujado por splatting de visibilidad sobre un z-buffer con relleno de huecos. Es el antecedente conceptual del disco de [5] y del surface splatting de las nubes de puntos (§2.7, `modules/render/src/PointRasterizer.cpp`). DOI 10.1145/344779.344936.

**[7] Moënne-Loccoz et al. 2024, 3D Gaussian Ray Tracing (3DGRT)** — Traza partículas Gaussianas con la BVH por hardware envolviendo cada una en un proxy (un icosaedro estirado), recoge los impactos en un k-buffer ordenado por tramos a lo largo del rayo y evalúa cada partícula en su respuesta máxima, lo que habilita rayos secundarios. La ruta *Hardware* de athenea usa los mismos proxies (`shaders/athenea/rt/rt_proxy.slang`) con `RayQuery` en línea (§2.6). Se aparta en el orden: registra hasta 256 entradas sin ordenar y las ordena tras la traversal con counting sort en 128 cubetas e inserción (de 2.52 s a 343 ms frente al k-buffer de 16), y en Metal `Auto` elige la ruta ComputeBvh (LBVH de Karras) porque cada candidato no opaco sale de la traversal por hardware (`modules/render/src/GaussianRayTracer.cpp`, `shaders/athenea/rt/rt_integrate.slang`). DOI 10.1145/3687934; arXiv 2407.07090.

**[8] Ren et al. 2024, Octree-GS** — Organiza anclas neuronales (al estilo de Scaffold-GS) en los niveles de un octree y elige el nivel de cada ancla por la distancia de la vista, con entrenamiento progresivo por niveles. Se cita como trabajo relacionado (§1.5); athenea no lo usa: su octree (§6.1) agrupa Gaussians explícitas por prefijos de Morton y las fusiona por momentos, sin anclas ni MLPs, y su LOD de producción es el de merge trees (§6.2). arXiv 2403.17898.

**[9] Xiong et al. 2026, NanoGS** — Simplificación sin entrenamiento que fusiona por pares Gaussians que son vecinas más cercanas mutuas bajo un coste, iterando. `athenea decimate --ratio`/`--tree` la ejecuta en el device (§6.2): cada nodo pesa 8 vecinos a cada lado a lo largo de dos órdenes de Morton (uno desplazado media cell), los pares que se eligen mutuamente se fusionan con un umbral de coste elegido por histograma hasta la mitad de los nodos por pasada, y se añaden guardas de pieza, vidrio, espejo, fold y color (§6.2.3) y la cima relajada (§6.2.4). Código: `shaders/athenea/lod/lod_pairs.slang`, `lod_pairs_final.slang`, `modules/lod/src/Lod.cpp`, `apps/athenea/src/CmdDecimate.cpp`. Sin datos bibliográficos completos en las fuentes; no se da identificador.

**[10] GaussianPOP** — Puntuación de poda que mide, por Gaussian, el error cuadrático que deja su eliminación en las vistas. athenea la acumula en la misma traversal de la mezcla, junto a la contribución $T\alpha$ (§2.5.2, `shaders/athenea/splat/splat_blend.slang`), y la usa `athenea decimate --prune-error` (`modules/usd/src/Decimate.cpp`); el manual (`docs/operations.es.md` §2.4.1) registra que en las piezas del Corvette deja huecos en superficies opacas, así que se conserva para nubes que no son superficies. Sin datos bibliográficos completos; no se da identificador.

**[11] Spark (World Labs)** — Renderer de 3DGS para la web sobre three.js, con su propio LoD (`LodSplat`) y un «núcleo más duro»: una opacidad $D>1$ se dibuja como $1-(1-g)^A$ con $A=\exp((D^2-1)/e)$. athenea adopta ese núcleo para merges de opacidad mayor que 1 (§2.5.2; `shaders/athenea/splat/frame.slang`, `splat_project.slang`, `lod_pairs_final.slang`, `--merge-opacity falloff` de `athenea decimate`). El player de §7 es el fork sparkwebGPU con backend WebGPU (demos públicas en https://sparkwebgpu.lucab.co.uk). URL github.com/sparkjsdev/spark.
:::

## 10.2 De meshes a Gaussians

12. Electronic Arts (SEED). mesh2splat: conversión de meshes en splats de Gaussians 3D por rasterización (código
    abierto, BSD-3). El origen de la conversión de la §3.
13. Mesh2GS. arXiv 2606.21898 (2026). El tamaño de cell derivado de una cámara de la §3.3.

::: {.nota}
**Notas a 10.2. De meshes a Gaussians** — Las dos obras que fijan la forma y el tamaño de cada Gaussian convertida.

**[12] Electronic Arts (SEED), mesh2splat** — Para cada triángulo, un shader de geometría lo proyecta sobre el plano de ejes hacia el que más apunta su normal (coordenadas «triplanares»), lo rasteriza a una resolución fija y cada texel cubierto emite una Gaussian tendida en el plano del triángulo, de tamaño el Jacobian de la inversa de la proyección por $\sigma=0.65$ y por $1/N$, con un grosor de $10^{-7}$ unidades, añadida con un contador atómico. athenea parte de él (§3.1, §3.4) pero lo reescribe como un kernel de cómputo dentro de un efecto AOFX (`plugins/mesh2splat/mesh2splat.slang`, `plugins/mesh2splat/Mesh2Splat.cpp`) con emisión determinista en tres pasadas (contar, explorar, escribir), $\sigma=1.0$ por defecto, grosor como fracción $\varphi$, la opacidad como coverage, la cota $\ell_{\max}$, `m2sOwnSize` para triángulos menores que una cell, aristas duras (`m2sHoldEdges`), Toksvig (`m2sSpread`) y captura de OpenPBR por capas. URL github.com/electronicarts/mesh2splat.

**[13] Mesh2GS** — Deriva el tamaño de una Gaussian del footprint de un píxel de una cámara, $h = z\,(A/P)/f$, con $z$ la distancia mínima a la caja dentro del frustum y sobre las muestras temporales de la cámara. athenea lo sigue con `--cell-from-camera CAM --camera-pixels P` (§3.3), pero toma la distancia euclídea del ojo a la caja acotada en `--time`, con mínimo $z_{\text{near}}$: igual para una caja visible y más fina, nunca más gruesa, para una al lado del frustum. Código: `shaders/athenea/usd/mesh2splat_cells.slang` y `apps/athenea/src/CmdMesh2Splat.cpp`. arXiv 2606.21898 (tal como lo cita el artículo).
:::

## 10.3 Transporte de luz, transfer precalculado y compresión

14. Sloan, P.-P., Kautz, J., Snyder, J. (2002). Precomputed Radiance Transfer for Real-Time Rendering in Dynamic,
    Low-Frequency Lighting Environments. SIGGRAPH.
15. Sloan, P.-P., Hall, J., Hart, J., Snyder, J. (2003). Clustered Principal Components for Precomputed Radiance
    Transfer. SIGGRAPH.
16. Sloan, P.-P. (2008). Stupid Spherical Harmonics (SH) Tricks. Game Developers Conference. (Ajuste de harmonics
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

::: {.nota}
**Notas a 10.3. Transporte de luz, transfer precalculado y compresión** — Las obras sobre las que se apoya el núcleo del baker (§4) y los códecs de la Parte II (§7).

**[14] Sloan, Kautz y Snyder 2002, Precomputed Radiance Transfer (PRT)** — Precalcula por punto un vector de transfer, la proyección sobre spherical harmonics de la visibilidad por el coseno (y, en su versión con interreflexiones, de la luz rebotada), $T_k=\frac1\pi\int V(\omega)\max(n\cdot\omega,0)Y_k(\omega)\,d\omega$; con el entorno proyectado en $L_k$, la radiancia difusa es el producto escalar $\sum_k T_k L_k$, y lo especular necesita una matriz de transfer. athenea hace el bake de exactamente ese $T_k$ con el path tracer en su variante `kTransfer` (§4.4.1, `modules/technique/src/PathTracer.cpp`, la variante declarada en `modules/technique/include/athenea/technique/PathTracer.h`), con las mitades directa e indirecta separadas y grado 3 (16 coeficientes); lo lee en `shaders/athenea/splat/splat_relight.slang`. Se aparta en lo especular: rechaza la matriz de transfer de 9 × 9 por Gaussian (§4.6.2) y usa en su lugar el reflected field y las open directions. DOI 10.1145/566654.566612.

**[15] Sloan et al. 2003, Clustered PCA (CPCA)** — Agrupa los vectores de transfer (cuantización vectorial) y dentro de cada cluster aplica PCA, $t\approx\mu_c+\sum_j w_j d_{c,j}$; al dibujar, la luz se proyecta una vez por cluster sobre su base, y cada punto solo hace un producto escalar con sus pesos. athenea lo usa dos veces (§7.2): la codificación 3 del fichero (`athc-convert --cpca`, una herramienta del player sparkwebGPU) y la base conjunta residente en la GPU del móvil, cuya «ruta proyectada» en el player es la proyección por cluster de Sloan. En el motor, `shaders/athenea/lod/transfer_codec.slang` (que cita la obra) y `athenea decimate --transfer-basis`. DOI 10.1145/882262.882281.

**[16] Sloan 2008, Stupid Spherical Harmonics (SH) Tricks** — Notas de GDC con, entre otras cosas, los harmonics zonales: una función simétrica alrededor de un eje $a$ se escribe $g_{lm}=z_l\,k_l\,Y_{lm}(a)$, $k_l=\sqrt{4\pi/(2l+1)}$, y gira girando su eje, sin matriz de rotación por banda; y el ajuste de lobes zonales a una función SH. athenea guarda dos lobes zonales por Gaussian en su propio marco (§4.4.5, §5.7), con $z_l=k_l\sum_m f_{lm}Y_{lm}(a)$ y residuo $\lVert f\rVert^2-\sum_l z_l^2$, ajustados en el efecto AOFX `plugins/splattransferzonal/splattransferzonal.slang` y leídos por `splatTransferFrame`.

**[17] Ramamoorthi y Hanrahan 2001** — Muestran que la irradiancia es la convolución del entorno con el coseno truncado, que en SH es diagonal, $E(n)=\sum_{lm}\hat A_l L_{lm}Y_{lm}(n)$ con $\hat A_0=\pi$, $\hat A_1=2\pi/3$, $\hat A_2=\pi/4$, y que como $\hat A_l$ decae deprisa nueve coeficientes la representan con poco error. athenea lee así la irradiancia de un cuerpo sin transfer (§4.3, `shaders/athenea/technique/env_project.slang`, `shaders/athenea/light/environment.slang`, ambos citan la obra) y usa los mismos pesos normalizados $\{1, 2/3, 1/4\}$ en el cociente $O_{\text{SH}}$ de §4.5. DOI 10.1145/383259.383317.

**[18] Křivánek y Colbert 2008, filtered importance sampling** — Para prefiltrar un entorno con pocas muestras, cada muestra de la BRDF lee el mapa en el nivel de mip cuyo texel tiene el ángulo sólido de la muestra, $\Omega_s=1/(Np)$, lo que quita el ruido sin más muestras. athenea lo aplica al construir el prefiltered map octaédrico de ocho niveles (§4.3, `shaders/athenea/technique/env_prefilter.slang`): cada muestra GGX lee un nivel más basto de lo que pide su ángulo sólido y nunca más estrecho que el texel que escribe, con 64 a 1024 muestras por nivel. DOI 10.1111/j.1467-8659.2008.01253.x.

**[19] Lagarde y de Rousiers 2014, Moving Frostbite to PBR** — Notas de curso que incluyen una aproximación de oclusión especular a partir de la oclusión ambiental, $\operatorname{sat}((n\cdot\omega_o+o)^{2^{-16r-1}}-1+o)$. El primer transfer de athenea la usaba con $o$ tomado de $T_0$ (§4.6.1); TX la sustituye preguntando al propio lobe por sus cells (`splatLobeOpen`, `splatConeOpen`) y la conserva solo como apertura del polish de materiales lisos ($\chi=0$, §4.6.3). Código: `shaders/athenea/splat/splat_relight.slang`.

**[20] Toksvig 2005, Mipmapping Normal Maps** — La longitud media de normales unitarias promediadas, $|\bar N|<1$, mide su dispersión, y el lobe especular se ensancha en consecuencia; en términos de microfacetas, la varianza de las normales se suma a la de las pendientes. athenea aplica la idea en la conversión, pero mide la dispersión geométricamente: la tasa de giro de las normales de vértice por el tamaño de la Gaussian, $s$, y $\alpha'^2=\alpha^2+2s^2$ (§3.7, §4.9.1; `m2sSpread`/`m2sWidened` en `plugins/mesh2splat/mesh2splat.slang`). Al dibujar lo deshace para los sharp lobes (`unwidened`, §4.9.2; `shaders/athenea/splat/splat_project.slang`, `splat_relight.slang`).

**[21] Veach y Guibas 1995, multiple importance sampling** — Combinar varias estrategias de muestreo con pesos $w_i=p_i^\beta/\sum_j p_j^\beta$ da un estimador insesgado de varianza cercana a la óptima; con $\beta=2$ es la heurística de potencia. El path tracer de athenea pondera así NEE frente al muestreo del material, $w_{\text{light}}=p_l^2/(p_l^2+p_m^2)$ con $p_l=P(k\mid x)\,p_k(\omega)$, y deja peso 1 a luces y lobes delta (§2.11). El kernel de path tracing lo genera `modules/technique/src/PathTracer.cpp`; el horno blanco destapó un peso de un solo lado. DOI 10.1145/218380.218498.

**[22] Conty Estevez y Kulla 2018, many lights** — Una BVH de luces cuyos nodos guardan caja, cono de orientación y potencia; la importancia de un nodo para un punto acota distancia y ángulo, y la traversal estocástica elige una luz con probabilidad conocida. athenea la construye con los mismos kernels de LBVH que la geometría y la usa con `athenea:chooseLights` (§2.10; `shaders/athenea/light/light_bvh.slang`, que cita el algoritmo 1 de unión de conos, y `light_bvh_build.slang`); su probabilidad entra en el MIS como $P(k\mid x)$. El artículo no describe la división adaptativa del árbol de la obra. DOI 10.1145/3233305.

**[23] Lloyd 1982; Arthur y Vassilvitskii 2007, k-means y k-means++** — Lloyd alterna asignar cada punto al centroide más cercano y recalcular cada centroide como media, lo que nunca sube la distorsión; k-means++ siembra muestreando con probabilidad proporcional a $D^2$, con garantía $O(\log k)$ en esperanza. athenea los usa en `fit-compact-basis` (semillas k-means++, Lloyd sobre todos los nodos, luego PCA por cluster; una herramienta del player sparkwebGPU, §7.2.2) y en el diccionario de open directions, un k-means en espacio de Hamming cuyo centroide es la mayoría bit a bit (`shaders/athenea/lod/shadow_vq.slang`, `athc-convert --shadow-vq`, §7.3). El codebook de oclusión de §6.7 es una variante con la restricción de contener (`shaders/athenea/lod/occlusion_mask.slang`). DOI (Lloyd) 10.1109/TIT.1982.1056489.
:::

## 10.4 Materiales y shading

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

::: {.nota}
**Notas a 10.4. Materiales y shading** — La biblioteca de lobes (`shaders/athenea/material/lobes.slang`, `microfacet.slang`) y los estándares de material que lee la conversión.

**[24] Walter et al. 2007, microfacetas con refracción** — Da la BTDF de un dieléctrico rugoso con el vector medio generalizado $h_t=-(\eta_i\,\omega_i+\eta_o\,\omega_o)$, $f_t=\frac{|\omega_i\cdot h||\omega_o\cdot h|}{|\omega_i\cdot n||\omega_o\cdot n|}\frac{\eta_o^2(1-F)GD}{(\eta_i(\omega_i\cdot h)+\eta_o(\omega_o\cdot h))^2}$, e introduce la distribución GGX. La transmisión de athenea sigue a Walter (§2.9): `transmissionNormal`, `microfacetTransmission` y `transmissionPdf` en `shaders/athenea/material/lobes.slang`, con el Fresnel de cada cara pagado por el propio lobe; GGX con $\alpha=r^2$ es la distribución de todo el motor (§4.1). DOI 10.2312/EGWR/EGSR07/195-206.

**[25] Heitz 2018, muestreo de normales visibles (VNDF)** — Muestrea la distribución de normales visibles desde $\omega_o$, $D_{\omega_o}(h)=G_1(\omega_o)\max(0,\omega_o\cdot h)D(h)/\cos\theta_o$, estirando el problema a la configuración isótropa de rugosidad 1 y muestreando un disco proyectado; reduce la varianza frente a muestrear $D$. Los lobes de microfacetas de athenea muestrean esta distribución (§2.9) y su densidad es la de Heitz (`ggxVisibleNormalPdf`), pero el muestreo usa la forma de casquete esférico de Dupuy y Benyoub 2023 (`ggxSampleVisibleNormal` en `shaders/athenea/material/microfacet.slang`), que da la misma distribución con menos aritmética. JCGT 7(4), jcgt.org/published/0007/04/01.

**[26] Oren y Nayar 1994; Portsmouth, Kutz y Hill 2025 (EON)** — Oren–Nayar modela una superficie difusa rugosa como cavidades en V lambertianas con pendientes de desviación $\sigma$, lo que oscurece el centro y aclara el borde frente a Lambert; EON es la variante de OpenPBR que conserva energía con un término de dispersión múltiple y tiene albedo y muestreo exactos. athenea tiene ambos: `kLobeOrenNayar` con `kFlagEnergyCompensation` llama a `orenNayarCompensated` (`shaders/athenea/material/lobes.slang`), descrito como «OpenPBR's energy-compensated Oren-Nayar (EON)» en `microfacet.slang` (§2.9). DOI (Oren–Nayar) 10.1145/192161.192213.

**[27] Burley 2012, Physically-Based Shading at Disney** — El modelo «principled»: un difuso con retrorreflexión $f_d=\frac\rho\pi(1+(F_{D90}-1)(1-\cos\theta_l)^5)(1+(F_{D90}-1)(1-\cos\theta_v)^5)$, $F_{D90}=0.5+2r\cos^2\theta_d$, GGX, Schlick y la convención $\alpha=r^2$ de roughness perceptual. athenea tiene el lobe `kLobeBurley` (`shaders/athenea/material/lobes.slang`) y el nodo MaterialX `shaders/athenea/material/mx/athenea_burley_diffuse_bsdf.slang`; la convención $\alpha=r^2$ es la de toda la §4.

**[28] Estevez y Kulla 2017; Zeltner, Burley y Chiang 2022, sheen** — Imageworks propone una distribución de microfibras $D(m)=(2+1/r)\sin^{1/r}\theta/(2\pi)$ para el brillo de los tejidos («Charlie sheen»); Zeltner et al. ajustan un sheen volumétrico con dispersión múltiple mediante linearly transformed cosines, con evaluación, muestreo y albedo baratos. La biblioteca tiene ambas formas (`imageworksSheenD`, `imageworksSheenDirectionalAlbedo`, `zeltnerSheenDirectionalAlbedo` en `shaders/athenea/material/microfacet.slang`; `mx/athenea_sheen_bsdf.slang`). En el relight, el sheen se responde como un cuerpo, su albedo direccional por $\mathbf B$ bajo el coat (§4.6.4, `splat_relight.slang`).

**[29] Lazarov 2013, Black Ops II** — Un ajuste analítico barato del término de entorno del split sum (el albedo direccional de la BRDF especular, DFG). athenea lo usó y lo sustituyó (§4.6.4): difería hasta un 76 % en rasante del `ggxDirectionalAlbedo` de MaterialX con compensación de dispersión múltiple, que es el que se usa ahora (`shaders/athenea/splat/splat_relight.slang`; la comprobación en `shaders/athenea/test/dfg_check.slang`).

**[30] Andersson, Kutz et al. 2024, OpenPBR Surface** — El uber-shader por capas de la ASWF (la fusión de Autodesk Standard Surface y Adobe Standard Material): base (metal como Schlick generalizado con tinte F82, dieléctrico, difuso EON, subsurface, transmisión), especular, coat con oscurecimiento de la base, fuzz (sheen), thin film, emisión, `geometry_opacity` y `geometry_thin_walled`. La conversión lee `open_pbr_surface` con sus nombres y defectos (§3.2, `modules/usd/src/MeshStage.cpp`), captura sus capas por Gaussian (§3.6), escribe el metal Schlick (`schlickMetal`), y el relight aplica el oscurecimiento del coat en $U$ (§4.6.4). Compilado a Slang ocupa 945 líneas y 55 palabras de blob (§2.9). URL github.com/AcademySoftwareFoundation/OpenPBR.

**[31] MaterialX 1.39** — Estándar de grafos de nodos con bibliotecas estándar (pbrlib: nodos BSDF, `layer`, `mix`) y generadores de shaders (GLSL, MSL, OSL, MDL y, desde 1.39, Slang). El `MaterialCompiler` de athenea deriva del `SlangShaderGenerator` de MaterialX 1.39.5 y cambia el significado de cuatro clases de nodo (superficie, BSDF/EDF, imagen, `heighttonormal`); se comprueba frente al genglsl de MaterialX a $3.2\cdot10^{-5}$ (§2.9). Código: `modules/material/src/MaterialCompiler.cpp`, `modules/technique/src/MaterialPrograms.cpp`, `shaders/athenea/material/mx/`. URL materialx.org.
:::

## 10.5 Geometría, estructuras de aceleración y LOD

32. Karras, T. (2012). Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees. High Performance
    Graphics.
33. Möller, T., Trumbore, B. (1997). Fast, Minimum Storage Ray–Triangle Intersection. Journal of Graphics Tools.
34. Runnalls, A. R. (2007). Kullback–Leibler Approach to Gaussian Mixture Reduction. IEEE Transactions on Aerospace
    and Electronic Systems.
35. Karis, B., Stubbe, R., Wihlidal, G. (2021). Nanite: A Deep Dive. SIGGRAPH Advances in Real-Time Rendering.

::: {.nota}
**Notas a 10.5. Geometría, estructuras de aceleración y LOD** — Las cuatro piezas algorítmicas que reutilizan la visibilidad de meshes, el ray tracer y los merge trees.

**[32] Karras 2012, construcción paralela de BVH** — Sobre claves de Morton ordenadas, cada nodo interno $i$ de un radix tree decide su dirección por el signo de $\delta(i,i+1)-\delta(i,i-1)$ ($\delta$ el prefijo común más largo), encuentra el otro extremo de su tramo por búsqueda binaria y parte en el bit más alto que difiere; todos los nodos en paralelo, $O(n)$, y las cajas suben con contadores atómicos. Es la LBVH de la ruta ComputeBvh del ray tracer de Gaussians (§2.6; `shaders/athenea/rt/bvh_hierarchy.slang`, `bvh_common.slang`, refit por niveles en `bvh_refit_level.slang`), de `VisibilityBvh` por mesh y sobre las instancias (§2.8; `modules/world/src/BvhScene.cpp`) y de la BVH de luces (§2.10). DOI 10.2312/EGGH/HPG12/033-037.

**[33] Möller y Trumbore 1997, intersección rayo–triángulo** — Resuelve $o+td=(1-u-v)a+ub+vc$ por la regla de Cramer con productos vectoriales ($P=d\times e_2$, $\det=e_1\cdot P$, $u=T\cdot P/\det$, $Q=T\times e_1$, $v=d\cdot Q/\det$, $t=e_2\cdot Q/\det$), sin guardar la ecuación del plano. athenea reconstruye el impacto a partir del buffer de visibilidad con él, en espacio de vista, para que las tres rutas sombreen igual (§2.8; `shaders/athenea/technique/surface.slang`), y lo usa en la traversal de `shaders/athenea/technique/visibility_bvh.slang`. DOI 10.1080/10867651.1997.10487468.

**[34] Runnalls 2007, reducción de mezclas de Gaussians** — Fusiona pares de componentes por momentos y elige el par con la menor cota superior de la divergencia de Kullback–Leibler, $B=\tfrac12[(w_i+w_j)\log\det\Sigma_{ij}-w_i\log\det\Sigma_i-w_j\log\det\Sigma_j]$. El término de forma del coste de merge de athenea es esa cota por unidad de masa (§6.2.2), con el segundo invariante $I_2$ (suma de productos de autovalores dos a dos) en lugar del determinante, para que un surfel sin volumen tenga coste finito (`shaders/athenea/lod/lod_pairs.slang`, que cita la obra). El player usa el mismo coste en `athc-convert --error`. DOI 10.1109/TAES.2007.4383588.

**[35] Karis, Stubbe y Wihlidal 2021, Nanite** — En un DAG de clusters simplificados, cada grupo lleva un error y una esfera forzados a ser monótonos (el padre con error mayor o igual que sus hijos y con esfera que contiene las suyas), de modo que el corte se decide por cluster en paralelo con la prueba «mi error proyectado $\le\tau<$ el de mi padre»; las pages de streaming se ordenan por ese error. athenea aplica exactamente esa construcción a los merge trees de Gaussians (§6.3, la proposición de una leaf dibujada una sola vez; `treeBoundsUp`, `treeErrorKeep` en `shaders/athenea/lod/lod_tree_cut.slang` y `modules/usd/src/Decimate.cpp`) y pagina por bandas del error del padre (§6.8, `shaders/athenea/lod/lod_pages.slang`; en el player, §7.5.1). URL advances.realtimerendering.com/s2021.
:::

## 10.6 Color

36. Academy of Motion Picture Arts and Sciences (2024). Transformaciones de salida de ACES 2.0 (referencia en CTL).
37. Hellwig, L., Stolitzka, D., Fairchild, M. D. (2022). El modelo de apariencia de color conocido como Hellwig 2022,
    una revisión de CAM16, que usan las transformaciones de salida de ACES 2.0.
38. Wrensch, B. Un ajuste analítico de la transformación de vista AgX. Nota en línea, tal como se cita en la §2.12.
39. Krystek, M. (1985). An Algorithm to Calculate Correlated Colour Temperature. Color Research and Application. (El
    ajuste del cuerpo negro de la §2.10.)
40. OpenColorIO. Academy Software Foundation.

::: {.nota}
**Notas a 10.6. Color** — La visualización (`technique::DisplayTransform`) y el color de las luces.

**[36] AMPAS 2024, transformaciones de salida de ACES 2.0** — Sustituyen las RRT/ODT de ACES 1 por una sola transformación parametrizada por la luminancia de pico de la pantalla: una escala tonal, compresión de croma y una compresión de gama hecha en el espacio de un modelo de apariencia ([37]), con su referencia en CTL. athenea la porta función a función a Slang (§2.12; `shaders/athenea/technique/aces2.slang`), con las tablas que la referencia calcula al iniciar (362 entradas por matiz) construidas en el device por `aces2_prepare.slang` y orquestadas por `modules/technique/src/Aces2.cpp`; los grises quedan neutros a $5\cdot10^{-6}$ y la ida y vuelta a $6\cdot10^{-6}$ del pico.

**[37] Hellwig, Stolitzka y Fairchild 2022** — Una revisión de CAM16 que simplifica el modelo y corrige su colorfulness y su croma, dando las coordenadas de apariencia $J$ (lightness), $M$ (colorfulness) y $h$ (hue). ACES 2.0 hace en ese espacio JMh su compresión de gama, y `aces2.slang` lo implementa «as the reference specialises it» (§2.12).

**[38] Wrensch, Minimal AgX** — AgX es la transformación de vista de Troy Sobotka; Wrensch publicó un ajuste analítico mínimo: una matriz de entrada, un log2 acotado a $[-12.47, 4.03]$ pasos y una sigmoide polinómica de sexto orden. athenea la ofrece como vista 1 de `shaders/athenea/technique/display.slang`, que cita el ajuste de 2023 (§2.12).

**[39] Krystek 1985, temperatura de color** — Un ajuste racional del lugar de Planck en coordenadas CIE 1960 $(u,v)$ en función de la temperatura, válido en el rango habitual de lámparas. athenea lo usa para el cuerpo negro de la temperatura de color de una luz UsdLux, convertido a sRGB lineal y normalizado a luminancia 1 (§2.10; `shaders/athenea/light/lights.slang`, «Krystek's rational fit of the Planckian locus»).

**[40] OpenColorIO (ASWF)** — Biblioteca y formato de configuración de gestión de color; su ruta de GPU genera el texto de un shader (GLSL, HLSL, MSL) y los valores de sus LUTs. athenea la usa como compilador (§2.12): `colour::ColourCompiler` pide el HLSL, lo carga como módulo Slang `athenea_cs_<hash>` y llena las LUTs en el device (`modules/colour/src/ColourCompiler.cpp`, `modules/technique/src/Ocio.cpp`); `colour::ColourNames` resuelve los espacios de color de entrada de las texturas (§2.9, `modules/colour/src/ColourNames.cpp`). URL opencolorio.org.
:::

## 10.7 Animación y visibilidad

41. Lin, et al. MLP de visibilidad direccional por parte del cuerpo para avatares articulados. Tal como se cita en la
    §5.8.
42. DNF-Avatar: un grid escalar de oclusión ambiental por hueso. Tal como se cita en la §5.8.

::: {.nota}
**Notas a 10.7. Animación y visibilidad** — Las dos ideas que la tabla direccional por parte de §5.8 combina. Ninguna de las dos tiene datos bibliográficos completos en las fuentes, así que no se da identificador.

**[41] Lin et al., MLP de visibilidad por parte** — Factorizan la visibilidad de un avatar articulado por parte del cuerpo, con una red direccional por parte que se mueve rígidamente con ella, de modo que la oclusión total es un producto que sobrevive a la pose. athenea toma la factorización y sustituye la red por una tabla: por parte, un grid de $N^3$ sondas con un mapa octaédrico de $O^2$ transmitancias, con bake una vez en la bind pose y leído por frame como $V_{i\ell}=\prod_{p\ne\pi(i)}F_p(T_p^{-1}x_i, L(T_p)^{-1}\omega_\ell)$. Código: `shaders/athenea/splat/splat_visibility.slang` (que cita a Lin et al. 2024), `splat_visibility_bake.slang`, `splat_visibility_read.slang`, `modules/technique/src/SplatVisibility.cpp` (`partitionJoints`, `visPartFrames`), CLI `athenea visibility`.

**[42] DNF-Avatar** — Hace el bake de la oclusión ambiental como un grid escalar por hueso. athenea conserva de ello el término ambiente de cada sonda (la media de la transmitancia sobre las direcciones), que es lo que se usa para los domes (§5.8, mismos ficheros); la dirección la aporta [41].
:::

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

::: {.nota}
**Notas a 10.8. Sistemas y estándares** — La infraestructura sobre la que corre todo, y dónde se engancha.

**[43] OpenUSD e Hydra (Pixar)** — USD compone una escena a partir de capas, referencias, payloads y variant sets, con esquemas como UsdGeom, UsdShade (bindings de material, `GeomSubset` con familia `materialBind`), UsdLux (luces), UsdSkel (skeletons, `SkelBindingAPI`, `UsdSkelBakeSkinning`) y `ParticleField`, cuyo `ParticleField3DGaussianSplat` es el prim del master. Hydra es la arquitectura de render: un render delegate recibe rprims, sprims y bprims, y los scene indices transforman la escena por el camino. athenea es el delegado `hdAthenea` (`modules/usd/src/RenderDelegate.cpp`; §2.13), lee `ParticleField` en `modules/usd/src/ParticleField.cpp`, añade el scene index `HdAtheneaSkelSplatSceneIndex` (`modules/usd/src/SkelSplat.cpp`, §5.5), y la conversión lee el stage directamente con UsdShade (`modules/usd/src/MeshStage.cpp`, §3.2). URL openusd.org.

**[44] He, Fatahalian y Foley 2018, Slang; slang-rhi** — Slang extiende HLSL con módulos, genéricos, interfaces, especialización en el enlace (`extern static const`) y una API de reflexión, y compila a SPIR-V, Metal, CUDA y WGSL; slang-rhi es su capa de hardware de render sobre Metal, Vulkan, CUDA y WebGPU. Todo kernel de athenea es Slang (§2.1), los parámetros se enlazan por nombre por reflexión, las constantes de enlace dan un programa por conjunto de valores y `DiskShaderCache` guarda un fichero por programa (§2.3; `modules/gpu`). El player sparkwebGPU recompila kernels de athenea a WGSL (§5.12, §7.8). DOI 10.1145/3197517.3201380.

**[45] Intel Open Image Denoise 2.5** — Denoiser de redes convolucionales (U-Net) para imágenes de path tracing, guiado opcionalmente por albedo y normal, con devices de GPU. `technique::Denoiser` ejecuta su filtro `RT` en la cola del motor (§2.11; `modules/technique/src/Denoiser.cpp`), compilado solo para GPU (regla 5 de §2.1), en Metal a través de buffers intermedios con seguimiento de riesgos y en Vulkan a través de CUDA sobre memoria importada. URL openimagedenoise.org.

**[46] W3C, WebGPU y WGSL** — La API de GPU de la web y su lenguaje de shaders, con límites que fijan varias decisiones del player: diez storage buffers por etapa en Chrome en un Mac (por eso el player reescribe `splatSkin` en un kernel propio, §5.12), la memoria de grupo de la pendiente del sharp lobe (§4.8.2), y la prohibición de enlazar un mismo buffer como lectura y escritura en un dispatch (§7.8). URLs w3.org/TR/webgpu y w3.org/TR/WGSL.

**[47] Friedman y Jones 2015, Cryptomatte** — Mattes automáticas por id: cada nombre de objeto se hashea (MurmurHash3 de 32 bits, con el exponente corregido para que el id sea un float válido), cada píxel guarda pares (id, coverage) ordenados por coverage en capas RGBA de dos rangos, y un manifest en los metadatos asocia nombres e ids. athenea calcula el id en `modules/core/src/Hash.cpp` (`murmurHash3x86_32`, `cryptomatteFixup`, `cryptomatteId`), acumula seis rangos en la mezcla (§2.5.2), guarda `cryptoObject` por Gaussian y `cryptoManifest` por nube, y lo usa para las máscaras de `--validate` (§3.10), los grupos de `--object-groups` (§6.2.5) y las capas `CryptoObject00`–`02` del nodo OpenFX (§2.14). URL github.com/Psyop/Cryptomatte.

**[48] O'Neill 2014, PCG; Halton 1960** — PCG avanza un generador congruencial lineal y aplica a la salida una permutación (xorshift y rotación dependiente del estado), con buena calidad estadística y estado pequeño; la secuencia de Halton es la inversa radical en bases primas, de baja discrepancia. El muestreador del path tracer pliega píxel, flujo, índice absoluto de camino, bounce y dimensión por un hash PCG (`shaders/athenea/material/sampling.slang`, «PCG (O'Neill)»), así que 256 paths en una pasada y 4 × 64 coinciden a relMSE $3.9\cdot10^{-15}$; entre pasadas la cámara se desplaza en el píxel con Halton (2, 3) (§2.11, `modules/usd/src/Engine.cpp`), y el player probó el mismo jitter en su antialiasing temporal (§5.12). URL pcg-random.org; DOI (Halton) 10.1007/BF01386213.
:::

# Apéndice A. Repositorios y comandos

## A.1 Código y demos

| repositorio | dónde | qué contiene |
|---|---|---|
| athenea | github.com/jesusluque/athenea | el motor y renderer de la §2: módulos, shaders, el delegado `hdAthenea`, la CLI `athenea`, el add-on de Blender `athenea_hydra` (`integrations/blender`, §2.14), el nodo OpenFX (`integrations/ofx`), los manuales `docs/operations.md` y `docs/development.md` con sus traducciones al español, y `docs/decisions.md` |
| sparkwebGPU | sparkwebgpu.lucab.co.uk | el player WebGPU de la §7, un fork de Spark: demos públicas del formato `.athc` v3 en el navegador |

::: {.nota}
**El código y las demos, en detalle** — Qué contiene el repositorio público y dónde se ve el player.

- **athenea** (github.com/jesusluque/athenea). Es el motor de la §2 con su CLI `athenea`, sus módulos (`modules/<name>`, bibliotecas estáticas `athenea::<name>`), sus shaders (`shaders/athenea/`), los bundles AOFX (`plugins/`: `mesh2splat`, `measure`, `splatbakefilter`, `splattransferzonal`, `invert`) y los manuales (`docs/operations.md`, `docs/development.md`, con sus traducciones `.es.md`, y `docs/decisions.md`). En el mismo repositorio están las dos integraciones: el add-on de Blender `athenea_hydra` (`integrations/blender/athenea_hydra`, §2.14) y el nodo OpenFX (`integrations/ofx`, compilado como `Athenea.ofx.bundle`). Las rutas de código de estas notas son relativas a este repositorio.
- **sparkwebGPU** (demos públicas en https://sparkwebgpu.lucab.co.uk). Es el player WebGPU de la §7, un fork de Spark: sus herramientas de la Parte II (`usd-athc`, `athc-convert`, `fit-compact-basis`, `floor_shadow`) derivan el formato `.athc` v3 del master, sus kernels de relight, de la ruta compacta, de la base CPCA conjunta y de skinning son kernels de athenea recompilados a WGSL, y el player los dibuja en el navegador.
:::

## A.2 Comandos y opciones

El manual de cada subcomando y opción es `docs/operations.md` (en español: `docs/operations.es.md`), que da el valor
por defecto, la unidad y la restricción de cada opción. Los comandos en los que se apoya este artículo:

| comando | se usa para | sección del manual | § |
|---|---|---|---|
| `athenea mesh2splat STAGE` (`--cell-from-camera`, `--camera-pixels`, `--transfer`, `--hide`, `--max-cells`, `--specular-filter`, `--edge-reach`, `--shadow-catcher`, `--light-catcher`, `--validate`, `--skinned`, `--lod-levels`) | conversión y bake | 2.8, 3.1 | 3, 4, 5 |
| `athenea decimate` (`--tree`, `--passes`, `--merge-glass`, `--object-groups`, `--tree-coverage`, `--tree-pages`, `--occlusion-mask`, `--occlusion-group`, `--cut-budget`, `--cut-cull-back`) | merge trees, cuts, pages, oclusión | 2.4.1, 3.3 | 5.9, 6 |
| `athenea visibility` | campos de visibilidad por parte | 2.9 | 5.8 |
| `athenea stage`, `athenea view` | tiempos (`--frames`) y el viewer | 2.5, 2.6 | 2 |
| `athenea compare` | comparación de imágenes en la GPU | 2.11 | 2.15 |
| `athenea aofx` | los efectos AOFX (mesh2splat, Measure) | 2.10, 7 | 2.14, 3.1 |
| `athc-convert` (`--gzip --planes`, `--shadow-vq`), `fit-compact-basis`, `floor_shadow --merge 2`, `usd-athc` | los formatos del player (sparkwebGPU) | — | 7 |
| `athenea-mcp`, herramienta `render` (`xform`, `raw`, `layers`, `sky`, `skyRotation`, `skyVisible`, `splatView`) | un frame para un compositor, el cielo cambiado, los canales de parámetros | 6 | 2.14, 4.3, 4.11 |
| `Athenea.ofx.bundle` (`ATHENEA_BUILD_OFX`) | el nodo OpenFX en Nuke y Natron | 6.1 | 2.14 |

::: {.nota}
**Cómo leer las notas de A.2** — Para cada comando de la tabla, una nota con cada opción listada: qué hace, su valor por defecto, su unidad y el código que la implementa. Los valores se toman de `docs/operations.es.md` (§2.4.1–§2.11, §6, §6.1) y de las declaraciones CLI11 (`add_option`/`add_flag`) en `apps/athenea/src/`; los de las herramientas del player, de las propias herramientas de sparkwebGPU.
:::

::: {.nota}
**`athenea mesh2splat STAGE`** — La conversión de §3 y el bake de §4–§5. Declaración de opciones en `apps/athenea/src/CmdMesh2Splat.cpp` (clase `Converter`); el kernel de emisión es el bundle AOFX `rt.sparrow.aofx.mesh2splat` (`plugins/mesh2splat/mesh2splat.slang`). La salida se escribe de forma atómica (fichero `.partial-<pid>` renombrado al terminar).

- `--cell-from-camera CAM` — ruta de un prim cámara; sin defecto. La cell de cada pieza es lo que cubre un píxel de esa cámara donde la caja del mesh le queda más cerca (como poco el plano cercano), en `--time` (§3.3). Sustituye a `--density`; `--cell-min`/`--cell-max` solo la acotan si se dan. Código: `shaders/athenea/usd/mesh2splat_cells.slang`.
- `--camera-pixels P` — entero de 1 a 65 536, por defecto 1920; píxeles a lo ancho de la apertura horizontal de la cámara. Los bakes del Corvette usaron 640, 1280 y hasta 5120 (§3.11).
- `--transfer` — flag, apagado. Hace el bake de un transfer TX (cuánto cielo llega a cada Gaussian) en vez de la luz que llegó (§4): 112 valores en grado 3 y 256 bits de open directions por defecto (`--transfer-degree` 3, `--transfer-cells` 16). Activa `--specular-filter 1`. Código: la variante `kTransfer` del path tracer (`modules/technique/src/PathTracer.cpp`, `modules/technique/include/athenea/technique/PathTracer.h`) por `StageRenderer::bakePointsOnDevice`, en rebanadas (`Converter::transfer`); el campo, en `shaders/athenea/usd/transfer_field_fill.slang` (`m2sFieldOverClosed`).
- `--hide P` — ruta de prim, repetible; ninguna por defecto. Escribe invisibilidad en `P` y todo lo que cuelga de él en una capa de sesión (el fichero no cambia), antes de la lectura. Solo afecta a la conversión: el bake traza el stage entero, así que una parte convertida sola conserva sombras e interreflexiones de las demás (§3.2, §3.11, §6.9).
- `--max-cells` — entero, por defecto 262 144 ($2^{18}$); cells como mucho que recorre un triángulo de su caja proyectada. Las que sobran se cuentan y se avisan; a 4736 px `Infinite_Black` perdió 359 628 y se rehízo con 1 048 576 (§3.3). Código: `plugins/mesh2splat/mesh2splat.slang`, `Mesh2Splat.cpp`.
- `--specular-filter k` — 0 a 4, sin unidad; por defecto 1 con `--transfer` y 0 en otro caso. Factor del ensanche de Toksvig: $s = k\max_e\angle/\lVert\Delta p\rVert\cdot\max(s_u,s_v)$, $\alpha'^2=\alpha^2+2s^2$ (§3.7). La variante B de §3.11 (0) muestra la línea oscura en el fold del faldón. Código: `m2sSpread`, `m2sWidened` en `plugins/mesh2splat/mesh2splat.slang`.
- `--edge-reach ρ` — 0 a 8, en tamaños propios de la Gaussian; por defecto 1.5; 0 no limita. El tamaño a través de una arista dura (abierta, no variedad o fold de más de 40°, `kCreaseDegrees`) se limita a $\max(d/\rho, 0.05r)$ (§3.5). Código: `m2sHoldEdges` en `plugins/mesh2splat/mesh2splat.slang`; aristas en `shaders/athenea/usd/mesh2splat_edges.slang`.
- `--shadow-catcher` — flag, apagado; necesita `--prim OBJETO`. Convierte la sombra que el objeto arroja sobre su suelo (encontrado o `--catcher-ground`) en un parche de Gaussians negras ensanchado `--catcher-margin` alturas (1.5) con cell `--catcher-cell` (0 = una centésima de la altura), con bake de transfer TX; implica `--transfer` (§3.9). Código: `modules/usd/src/ShadowCatcher.cpp`.
- `--light-catcher` — flag, apagado; incompatible con `--transfer` y `--shadow-catcher`. El mismo parche, alargado `--light-reach` longitudes del objeto (3), con bake de radiancia por grupo de luces con la cúpula en negro y dibujado de forma aditiva (§3.9). Escribe `primvars:athenea:splat:lightCatcher`.
- `--validate DIR` — directorio; sin defecto. Mide la conversión material a material frente al path trace del stage (`--validate-paths` 512, `--validate-bounces` 6, `--validate-size` 960 × 540, `--validate-sky` las luces de la escena o `white` o una imagen), con máscaras por Cryptomatte y el efecto Measure, en `validate.json` (§3.10). Código: `apps/athenea/src/Mesh2SplatValidate.cpp`, `plugins/measure/measure.slang`.
- `--skinned` — flag, apagado. Lee la bind pose y conserva el skeleton: cuatro joints, sus pesos y los gradientes por Gaussian; obliga a `--no-bake`, guarda un `--transfer` como dos lobes zonales trazados en la bind pose (`StageRenderer::holdBindPose`), y escribe `skinningXforms` en `--range` (§5.2–§5.7). Código: `blendInfluences` en `plugins/mesh2splat/mesh2splat.slang`, el ajuste zonal en `plugins/splattransferzonal/`. No admite salida `.athc`.
- `--lod-levels N` — entero, por defecto 1 (ninguno). Convierte otra vez a la mitad de resolución en cada nivel, cada uno un `<nombre>_lod<n>.usdc` con su propio rig, y `-o` pasa a ser la escena que los dibuja como una nube (§5.9). El motor elige el nivel con `Engine::lodLevelsFor` (umbral `athenea:lod:threshold`, 1 px) y no posa los no dibujados (`deferredPose`; `modules/usd/src/Engine.cpp`).
:::

::: {.nota}
**`athenea decimate`** — Merge trees, cuts, pages y oclusión (§6). Opciones en `apps/athenea/src/CmdDecimate.cpp`; el trabajo en `usd::decimateStage` (`modules/usd/src/Decimate.cpp`), `modules/usd/src/Occlusion.cpp` y los kernels `shaders/athenea/lod/lod_pairs.slang`, `lod_pairs_final.slang`, `lod_tree_cut.slang`, `lod_pages.slang`, `occlusion_mask.slang`. Con `--ratio`, la fusión es por pares (NanoGS); sin él, se corta el octree de §6.1.

- `--tree RUTA` — con `--ratio`: escribe además el merge tree entero, las leaves en orden de profundidad y después cada merge, con `lodParent`, `lodStart`, `lodSize`, `lodCost` y la constante `lodTree` (§6.2.4). Sin defecto.
- `--passes N` — con `--ratio`: como mucho N pasadas por pares; por defecto 60. Cada pasada fusiona como mucho la mitad de lo que queda, así que un grupo de muchas piezas pequeñas necesita más: 60 dejaban 116 549 raíces de 323 613 leaves y 1000 llegaron a 1 298 en 310 pasadas; el Corvette entero usó 4000 (§6.2.5).
- `--merge-glass` — flag, apagado; con `--ratio`. Las Gaussians que transmiten también se fusionan entre sí; apagado, el vidrio se conserva tal cual (raíces igual a leaves). La guarda vidrio con no vidrio se mantiene siempre (§6.2.3, §6.2.5).
- `--object-groups K` y `--object-group i` — enteros; apagado por defecto. Solo se procesa el grupo de piezas cuyo `cryptoObject` módulo K es i, para un asset que no cabe en una pasada; como los merges no cruzan piezas, el tree de cada grupo es el subárbol exacto del asset entero (§6.2.5).
- `--tree-coverage` — flag, apagado; la entrada es un merge tree. La opacidad de cada merge pasa a ser lo que sus leaves cubren de verdad, el ajuste por mínimos cuadrados $a^\star=\sum GC/\sum G^2$ sobre una cuadrícula de 8 × 8 en $[-1.25,1.25]^2$ (§6.4), con suelo $2/255$; la masa queda en `coverageMass` y el spill en `lodSpill`. Código: `treeCoverageToFile` en `lod_tree_cut.slang`.
- `--tree-pages P` — número de Gaussians por page; apagado por defecto (el artículo usa 3000). Escribe `lodPage`, `lodPageInfo` y `lodPageBounds` ordenando los nodos por banda del error del padre, primera leaf y tamaño (§6.8). Código: `shaders/athenea/lod/lod_pages.slang`.
- `--occlusion-mask N` — entradas del codebook; apagado. La entrada debe ser un tree con bits de sombra de 16 × 16: escribe la máscara conservadora de cada nodo como índice a un codebook de N entradas de 256 bits (8192 en el coche a 640 px, 8182 en el close-up level), §6.7. Código: `shaders/athenea/lod/occlusion_mask.slang`, `modules/usd/src/Occlusion.cpp`.
- `--occlusion-group RUTAS` — con `--occlusion-mask`: más trees que comparten el codebook de la entrada, repartido por necesidad; un tree cuya salida es su misma ruta se escribe al lado y se renombra encima (`occlusionMasksShared`, §6.9).
- `--cut-budget N` — Gaussians; apagado. Con un tree de entrada, conserva el cut que se dibujaría desde `--camera` con como mucho N nodos, bisecando $\log\tau$ en $[10^{-4},10^4]$ en 40 pasos (§6.3).
- `--cut-cull-back` — flag, apagado; necesita `--cut-bounds`. Deja fuera los nodos cuyo cono de normales entero mira en sentido contrario a la cámara, solo en closed shells que no son vidrio ($\theta<\pi/2\wedge n\cdot v<-\sin\theta$, §6.6). Código: `treeCulledMark` en `lod_tree_cut.slang`.
:::

::: {.nota}
**`athenea visibility`** — El bake de los campos de visibilidad por parte de §5.8, sobre una nube con skinning ya convertida. Opciones en `apps/athenea/src/CmdVisibility.cpp`; la partición (`partitionJoints`) y la lectura (`visPartFrames`) en `modules/technique/src/SplatVisibility.cpp`; el bake en `shaders/athenea/splat/splat_visibility_bake.slang`, con rayos inline (por eso no en CUDA). La tabla no lista opciones; las que fijan las cifras de §5.8 son `--parts` (12 partes), `--min-joints` (6; un subárbol menor se queda con la parte de su padre), `--grid` (24 sondas por eje de la caja de una parte), `--octave` (16 direcciones por lado del mapa octaédrico) y `--cut` (0.001, transmitancia bajo la que se detiene un rayo del bake). Son obligatorios `--skeleton-stage` y `--skeleton-prim` (la jerarquía de joints); `--prim` es por defecto `/World/Splats`, y sin `-o` edita el fichero en su sitio. En el gorrión: 22 s y 81 MB de campos.
:::

::: {.nota}
**`athenea stage`, `athenea view` (`--frames`)** — `--frames N` es el número de frames que se repiten para medir. En `athenea stage` (`apps/athenea/src/CmdStage.cpp`) vale 1 por defecto e informa del primero, la mediana y el más rápido; en `athenea view` (`apps/athenea/src/CmdView.cpp`, la ventana en `modules/view/src/Viewer.cpp`) vale 0, que corre hasta cerrar la ventana, y con N cierra tras N frames e imprime sus tiempos. Las tablas del artículo citan la mediana de los frames tras el primero, en milisegundos y en el preset release (§2.8, §2.10, §2.11, §5.6). El resto de opciones que pesan en las medidas: `--technique` (`raster` o `rt`), `--visibility` (`automatic`, `raster`, `rays`, `bvh`), `--size` (stage 1920x1080; view 1600x900 en puntos), `--path-samples`/`--path-bounces`/`--path-total` (stage 1/1/1; view 1/4/64).
:::

::: {.nota}
**`athenea compare`** — La comparación de imágenes de §2.15 como comando: con una imagen da medias y máximos por canal; con dos (`image` y `reference`, EXR del mismo tamaño) da el relMSE $\overline{(a-b)^2/(b^2+10^{-2})}$, el p99 y la mayor diferencia relativa, la distribución en códigos sRGB de 8 bits (p99, máximo, píxeles por encima de 2), el PSNR del color codificado y la SSIM media de luminancia en caja de 7 × 7. Opciones: `--window X0 Y0 X1 Y1` (píxeles, filas desde abajo; solo afecta a las medias), `--heatmap RUTA` (EXR del efecto), `--show` (`source`, `difference`, `relative`, `codes`; por defecto `codes`), `--gain` (≥ 0, por defecto 1, solo la imagen), `--path`. Cada número lo calcula un kernel del efecto Measure `rt.sparrow.aofx.measure` (`plugins/measure/measure.slang`), corrido por el host AOFX (`apps/athenea/src/CmdCompare.cpp`).
:::

::: {.nota}
**`athenea aofx`** — Dos subcomandos (`apps/athenea/src/CmdAofx.cpp`) sobre el host compartido de aopenfx, del que el motor solo declara lo que aporta (`modules/aofx/host/src/Host.cpp`, §2.14). `athenea aofx list` imprime cada bundle encontrado, cargado o rechazado, con sus efectos, entradas y parámetros; `athenea aofx run EFFECT INPUTS…` ejecuta un efecto sobre EXR, con `-o` (por defecto `out.exr`), `--param nombre=valor` repetible, `--time` (tiempo de frame, 0) y `--path` (directorios buscados después de `$AOFX_PLUGIN_PATH`). Los bundles del artículo son `rt.sparrow.aofx.mesh2splat` (§3.1), `rt.sparrow.aofx.measure` (§2.15, §3.10), `rt.sparrow.aofx.splatbakefilter` (el filtro à-trous de §4.4.4) y `rt.sparrow.aofx.splattransferzonal` (§4.4.5).
:::

::: {.nota}
**Herramientas del player (sparkwebGPU)** — Ninguna es del motor; derivan el formato web del master (Parte II).

- `usd-athc` — lee el master USDC con el crate `openusd` y escribe un `.athc` v3 por secciones, transfer incluido (`--transfer full|112|84|…|none`); con `--tree` lee el merge tree de `athenea decimate --tree` y conserva sus nodos, y con `--lod-errors` escribe la sección `LODE` ($\varepsilon$, $\rho$, cono). `--add` une varias nubes en un fichero y `--only-prim`/`--exclude-prim` filtran por el `cryptoManifest` (§7, paso 1; §5.12 para las nubes con skinning).
- `athc-convert --gzip --planes` — `--gzip` comprime cada sección; `--planes` elige por sección la codificación más pequeña entre 0 (palabras), 1 (planos de bytes) y 2 (planos de diferencias) (§7.1: Corvette ligero −10.6 %, HD −13.9 %).
- `athc-convert --shadow-vq C[,R[,P]]` — los 256 bits de open directions como diccionario: k-means en espacio de Hamming con centroides por mayoría, C centroides (1024), R rondas (12), piezas de P splats (16 384), e imprime las distancias de Hamming p50/p90/p99 (§7.3). El diccionario compartido de 4096 entradas del asset es el de por defecto en la ruta compacta.
- `fit-compact-basis` — ajusta la base CPCA conjunta del tier compacto sobre las piezas de un asset: `--clusters` (256; 512 en v8), `--iters` (16 rondas de Lloyd), `--seed` (1), `--seed-sample` (200 000), `--sample S` (un nodo de cada S, para el coche entero), `--held-out` (piezas medidas sin ajustar); escribe `<out>.json` y `<out>.bin`, que lee `usd-athc --compact-basis` (§7.2.2).
- `floor_shadow --merge 2` — convierte el `.athc` de un shadow catcher en el fichero `FSH1` de la contact shadow; `--merge N` fusiona por momentos grupos de N × N vecinos (centro y covarianza ponderados por opacidad × área, masa conservada, cells por mayoría): 309 k Gaussians a 78 k (§7.9).
:::

::: {.nota}
**`athenea-mcp`, herramienta `render`** — El motor como servidor MCP (JSON-RPC 2.0 sobre stdio; `apps/athenea-mcp/main.cpp`, herramientas en `modules/mcp/src/Tools.cpp`), con el device y el stage calientes entre llamadas (§2.14). Los argumentos de `render` que usa el artículo:

- `xform` — dieciséis números, la matriz de cámara a mundo fila a fila, mirando hacia −z con +y arriba; con `focal`, `haperture`, `vaperture` en milímetros y `near`/`far` en unidades de la escena. Es la cámara tal como la guarda un compositor.
- `raw` — ruta de un fichero que recibe los números del frame en lugar de una imagen: RGBA como cuatro float32 por píxel y luego la profundidad de vista (0 donde no se dibujó nada), fila inferior primero, lineal y premultiplicado; la respuesta es solo texto. Es lo que pide el nodo OpenFX.
- `layers` — lista; añade tras la profundidad cuatro floats por píxel de `P` (posición en el mundo, que pide una cámara propia, `xform` o `eye`) y de `CryptoObject00`–`02` (id, coverage, id, coverage); un frame con matte responde además el manifest.
- `sky` — una imagen `.hdr` o `.exr` lat-long en todos los domos (vacío devuelve la del stage); `skyRotation` — grados alrededor del eje vertical de la escena; `skyVisible` — booleano; a `false` el fondo sale transparente mientras el cielo sigue iluminando (§4.3: una nube girada lee su transfer en su propio sistema con el cielo girado a él).
- `splatView` — el canal de parámetros de §4.11 que se dibuja en lugar de la luz (`albedo`, `roughness`, `metallic`, `transmission`, `ior`, `normal`, `opacity`, `size`, `skyVisibility`, `bounce`, `openCells`, `emission`, `field`, `baked`, `materialRoughness`; `light` vuelve a la luz), el ajuste `athenea:splatView`.
:::

::: {.nota}
**`Athenea.ofx.bundle` (`ATHENEA_BUILD_OFX`)** — `ATHENEA_BUILD_OFX` es una opción de CMake, activada por defecto (`CMakeLists.txt`), que compila en macOS y Linux (no en Windows) `<build>/ofx/Athenea.ofx.bundle` a partir de `integrations/ofx/src` (`AtheneaOfx.cpp`, `RenderServer.cpp`) sobre el OpenFX de Natron. El nodo no enlaza el motor: arranca `athenea-mcp` como proceso hijo (el del bundle en `Contents/Resources/bin`, u otro nombrado por la variable `ATHENEA_MCP`) y lee cada frame de un fichero temporal con `raw`. Sus parámetros: `cloud` (la nube), `translate` (0 0 5), `rotate` (grados) con `rot_order` (ZXY, el de Nuke), `focal` (50 mm), `haperture` (24.576 mm), `vaperture` (0 = ajustada al formato), `near`/`far` (0.1/10000), `time_offset` (frames), `sky`, `sky_rotation` (grados), `sky_visible` y `output` (`rgba`, `depth`, `P`, `CryptoObject00`–`02`, la capa que muestra un host sin planos). Se instala en `/Library/OFX/Plugins` o en `OFX_PLUGIN_PATH` (§2.14, manual §6.1).
:::

Las variables de entorno (`ATHENEA_GPU_BUDGET`, `ATHENEA_STAGES`) se enumeran en la §8.1 de `docs/operations.md`, los
formatos (`.athc`, los esquemas sin código) en la §8.2 y la §4.3.

::: {.nota}
**Variables de entorno y formatos** — `ATHENEA_GPU_BUDGET` es la memoria del device que puede ocupar una ejecución, en MiB; sin ella el budget es el `recommendedMaxWorkingSetSize` de Metal (18 186 MiB en el M5 Pro) y en CUDA y Vulkan no hay ninguno. `Device::admit` rechaza toda reserva que lo pase, y en Apple silicon una reserva debe caber además en la memoria física libre menos 1.5 GiB (§2.3; `modules/gpu/src/Device.cpp`). `ATHENEA_STAGES=1` imprime una línea por frame con el tiempo de cada etapa del raster de splats (commit, visibilidad por partes, proyección, recuentos, sort por profundidad, emisión, sort por tile, blend) y de la capa de meshes, esperando a cada una, así que el frame es más lento (§2.5.3; `modules/usd/src/Engine.cpp`). En el manual actual, la §8.1 remite a la §1.2 de `docs/operations.es.md` (para `ATHENEA_GPU_BUDGET`) y a la §3 de `docs/development.es.md` (para `ATHENEA_STAGES`); los esquemas sin código (`AtheneaSplatLightingAPI`, `AtheneaStreamedAssetAPI` y otros) están en la §4.3 y los formatos en la §8.2.
:::

# Apéndice B. Glosario {.unnumbered}

Términos técnicos del artículo, por orden alfabético, con la sección donde aparecen. Las entradas conservan el término en inglés que usa el texto.

à-trous filter
: Filtro de bake de splats que promedia vecinos con huecos que se doblan en cada iteración, ponderados por posición en el plano tangente, orientación e id de prim. Filtra la indirect half y el reflected field, nunca la direct half (§4.4.4, `--bake-filter`, 3 iteraciones).

Adaptive sampling
: Muestreo adaptativo del path tracer: un píxel se detiene cuando el error relativo estimado de su media cae por debajo de $\varepsilon$, usando la mayor entre su varianza y la media de la de sus vecinos 3 × 3 (§2.11).

Affine approximation (EWA)
: Aproximación de la proyección perspectiva por su Jacobian en el centro de la Gaussian, que convierte una Gaussian 3D en una 2D. Es exacta en ortográfica y es la fuente de la diferencia entre raster y ray tracer (§2.5.1, §2.15).

AgX
: Transformación de vista de Troy Sobotka, usada en el ajuste analítico de Wrensch: log2 sobre $[-12.47, 4.03]$ pasos y una sigmoide de sexto orden (§2.12).

Albedo
: Fracción de luz que refleja una superficie difusa, el color base sin luz; $\rho$ en la notación de §4.1 y canal `albedo` de §4.11.

Alpha compositing
: Composición de delante hacia atrás $C=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$, el operador over; el rasterizer, la referencia y el ray tracer coinciden con su forma analítica a p99 0 (§2.4, §2.5.2).

Ambient occlusion
: Fracción del hemisferio desde la que llega cielo; en athenea sale del término constante del transfer, $o=\operatorname{sat}(T_0/0.28209)$, y es el canal `skyVisibility` (§4.5, §4.11).

Anisotropic Gaussian
: Gaussian 3D con tres escalas distintas a lo largo de ejes girados, $\Sigma=RSS^\top R^\top$; la primitiva de todo el sistema (§1.1, §2.4).

Antialiasing filter
: Dilatación de la covarianza en pantalla en 0.3 px², con la opacidad escalada por $k=\sqrt{\det\Sigma/\det\Sigma'}$ para conservar la energía (§2.5.1, §7.7).

Any-hit ray
: Rayo que acepta cualquier intersección y no la más cercana; el bake de open directions traza uno por cell para saber si la dirección sale de la escena (§4.4.2).

AOFX
: API de efectos de aopenfx (SDK `sdk/` y host `host/`), con ABI que solo cambia de forma aditiva; mesh2splat, Measure y el filtro de bake son bundles AOFX (§2.14, §3.1).

AOV
: Arbitrary output variable: una salida del render distinta del color (profundidad, ids, normal, Cryptomatte), convertida en el device solo cuando un host mapea el buffer (§2.13).

Argument buffer
: Mecanismo de Metal para pasar muchos recursos como una tabla; el `ParameterBlock` de 1024 texturas del `TextureStore` se enlaza así (§2.9).

Back-face culling
: Descarte de los nodos cuyo cono de normales entero mira hacia atrás, solo en closed shells que no son vidrio; sube la puerta un 12 % y quita la red de grietas de las dos pieles (§6.6, §7.5.1).

Bake
: Cálculo previo, una vez, de algo que el dibujo solo lee: aquí, el transporte de luz en cada Gaussian con el path tracer (§4) o los campos de visibilidad por parte (§5.8).

Band (error band)
: Octava del error del padre de un nodo, $b(n)=\operatorname{clamp}(\lfloor\log_2\varepsilon(\mathrm{par}(n))\rfloor+40,0,63)$; las pages se ordenan por banda (§6.8, §7.5.1).

Barycentric coordinates
: Pesos $(\lambda_0,\lambda_1,\lambda_2)$ que expresan un punto del triángulo como combinación de sus vértices; sitúan cada Gaussian emitida y mezclan las influencias de joint (§3.4, §5.3).

Bind pose
: La pose en que se ligó la geometría al skeleton; la nube con skinning se guarda en ella y el bake del transfer y de la visibilidad se hace en ella (§5.1, §5.7).

BLAS / TLAS
: Bottom- y top-level acceleration structures del ray tracing por hardware: una por mesh y una por frame sobre las instancias (§2.8).

Blend (stage)
: Última etapa del tile rasterizer: un grupo de 16 × 16 hilos por tile recorre sus pares de delante hacia atrás y compone cada píxel (§2.5.2).

Body (cuerpo)
: La parte del relight que no depende del ojo, $\mathbf B=\sum_k L_k(T_k^{\text{dir}}+T_k^{\text{ind}})$ más el sol y su bounce; se guarda en caché mientras no cambian cielo ni luces (§4.5).

Bounce
: Rebote de un camino tras el primer impacto; también el canal que muestra la indirect half bajo un cielo blanco (§2.11, §4.11).

Bounding sphere
: Esfera $\rho(n)$ de cada nodo que contiene las de sus hijos; con el error monótono hace que el cut sea una sola prueba paralela (§6.3).

BSDF
: Bidirectional scattering distribution function: cómo una superficie dispersa la luz entre dos direcciones; en athenea un nodo BSDF de MaterialX empuja lobes a una `LobeStack` (§2.9).

BVH
: Bounding volume hierarchy: árbol de cajas para trazar rayos; athenea usa la del hardware y una LBVH de cómputo (§2.6, §2.8).

Byte planes
: Codificación sin pérdida que transpone un array en planos (el byte 0 de todos los elementos, luego el 1…) para que gzip encuentre tramos; codificación 1 del `.athc` v3 (§7.1).

Car crop
: Recorte de 900 × 420 del coche en el frame de 1920 × 1080 sobre el que se mide el PSNR del coche (§1.6).

Cell
: La longitud $h$ en el mundo que representa una Gaussian convertida; sale del modelo, del mesh o del footprint de un píxel de una cámara (§3.3). También, una dirección del grid octaédrico de open directions (§4.4.2).

Chi-square test (χ²)
: Prueba estadística que compara histogramas de direcciones muestreadas con la pdf integrada por intervalo; valida lobes y luces (§2.9, §2.10).

Chunk
: Bloque de 65 536 splats consecutivos en orden de Morton del formato `.athc` original, la unidad de streaming del LOD de §6.1.

Clear coat
: Capa dieléctrica fina sobre la base (laca); en la pintura del Corvette es lo que el ojo ve, con peso 1, roughness 0 e índice 1.45 (§3.6, §4.6.3).

Clip
: Una animación del skeleton; en el gorrión cada clip es una capa que solo cambia `skinningXforms`, ofrecida como variante (§5.5).

Close-up level
: El bake del Corvette a 5120 px para vistas cercanas: 24 merge trees, 74.5 M leaves, 87.4 GB (§3.11, §6.9).

Closed shell
: Mesh cerrado por la regla $n_{\text{open}}<2\sqrt{1.5T}$; sus Gaussians llevan `closed = 1` y admiten back-face culling (§3.5, §6.6).

Clustered PCA (CPCA)
: Compresión que agrupa vectores y aplica PCA dentro de cada cluster; comprime el transfer en el fichero y en la GPU del móvil (§7.2).

Codebook
: Diccionario de entradas al que un índice sustituye un dato; el de oclusión es conservador (§6.7) y el de open directions, por mayoría (§7.3).

Command buffer
: Lista de trabajo enviada a la GPU; un fallo de memoria en uno se devuelve ahora como error en el siguiente `submit()` (§2.3).

Compact record
: Registro de 40 bytes por Gaussian en planos de bytes del player: posición, albedo, opacidad, escala, rotación, normal, curvatura, material, cluster, pesos y sombra (§7.4).

Compute kernel
: Programa de GPU de propósito general lanzado en hilos; toda aritmética sobre datos de la escena es uno (§2.1, §2.3).

Conic
: La inversa de la covarianza en pantalla, $Q=\Sigma'^{-1}$, que evalúa la Gaussian en cada píxel (§2.5.2).

Contact shadow
: Pase del player que splatea las Gaussians negras del shadow catcher sobre la cúpula antes del coche (§7.9).

Counting sort
: Ordenación por recuento en cubetas; el ray tracer ordena sus 256 entradas en 128 cubetas (§2.6) y el shade del player agrupa por cluster (§7.8).

Covariance
: Matriz $\Sigma$ que da forma y orientación a una Gaussian; en pantalla, $\Sigma_{2D}$ (§2.5.1).

Coverage
: Fracción de lo que hay detrás que cubre una superficie; la conversión trata toda opacidad como coverage y la traduce a $\alpha$ por Gaussian (§3.4), y un merge recibe la suya por mínimos cuadrados (§6.4).

Crate
: El formato binario de USD (`.usdc`); sus arrays comprimidos llegan en buffers nuevos en cada lectura, de ahí `_held` (§5.6).

Cryptomatte
: Estándar de mattes por id: hash del nombre, pares (id, coverage) por píxel y manifest; athenea acumula seis rangos y guarda `cryptoObject` por Gaussian (§2.5.2, §2.14).

Curvature
: Los tres valores $(S_{uu},S_{uv},S_{vv})$ del shape operator guardados por Gaussian (§4.2, §4.8).

Cut
: El conjunto de nodos de un merge tree que se dibujan para una vista (§1.6, §6.3).

Decimation
: Reducir el número de Gaussians fusionándolas; la del gorrión conserva el rig (§5.9) y la de §6 construye merge trees.

Delta light
: Luz o lobe con densidad concentrada en una dirección (puntual, espejo perfecto), que no puede alcanzarse por muestreo del otro lado y conserva peso 1 en el MIS (§2.11).

Denoiser
: Filtro que quita el ruido de un path trace; athenea usa el `RT` de OIDN guiado por albedo y normal (§2.11).

Depth key
: Clave de sort de 24 bits `bit_cast<uint>(z) >> 7`, que conserva el orden de floats positivos (§2.5.1).

Device
: La GPU tal como la ve slang-rhi; «en el device» es en la GPU (§2.1).

Dielectric
: Material no conductor (vidrio, laca) cuyo Fresnel depende del índice de refracción (§2.9, §4.6.4).

Direct half
: Parte del transfer de los caminos que escapan justo tras el primer vértice: escalar, solo geometría, calculada por cuadratura en TX (§4.4.1, §4.4.2).

Dispatch
: Lanzamiento de un kernel de cómputo; athenea despacha en hilos, redondeando los grupos hacia arriba (§2.3).

Dome light
: Luz de entorno UsdLux con una imagen lat-long; se muestrea con peso coseno o deformando su cadena de mips (§2.10, §4.3).

Dual quaternion skinning
: Skinning que mezcla cuaterniones duales en vez de matrices; no implementado para nubes (§5.5, §5.13).

Edge reach
: Cuánto puede asomar una Gaussian más allá de una arista dura, en sus propios tamaños; 1.5 por defecto (§3.5).

Eigen decomposition (Jacobi)
: Diagonalización de una covarianza fusionada por barridos de Jacobi en el shader, con el eje más delgado en tercer lugar (§6.1, §6.2.1).

Elastic term
: Segundo término del Jacobian del skinning, $\sum_j(X_jq)\nabla w_j^\top$, que estira una Gaussian donde los pesos varían (§5.4.2).

Emissive table
: Tabla de triángulos emisores ponderados por área y luminancia que el path tracer muestrea como luces (§2.11).

Environment albedo
: Integral de la BRDF especular sobre el hemisferio (DFG, término del split sum), $P$ en el polish; athenea usa el `ggxDirectionalAlbedo` de MaterialX (§4.6.4).

EWA splatting
: Elliptical weighted average: proyección de una Gaussian por la aproximación afín local de la perspectiva (§1.5, §2.5.1).

EXR
: OpenEXR, formato de imagen float con canales y capas arbitrarios; salida lineal premultiplicada del motor (§2.14).

Eye-dome lighting
: Sombreado en espacio de pantalla de nubes de puntos a partir de la profundidad (§2.7).

Filtered importance sampling
: Lectura de cada muestra en el nivel de mip de su ángulo sólido; construye el prefiltered map (§4.3).

Fold
: Arista donde dos triángulos giran más de 40°; una arista dura (§3.5), y donde el giro por píxel se apaga y el lobe se ensancha (§4.9.4).

Footprint
: Lo que cubre una Gaussian en pantalla o un píxel en el mundo; fija el tamaño de cell (§3.3) y el número de tiles (§2.5.1).

Fresnel
: Fracción de luz reflejada en una interfaz según el ángulo; athenea usa el exacto para el coat en vez de Schlick (§4.6.4).

Frustum
: Pirámide truncada visible de la cámara; el centro de un splat se acota al frustum ensanchado a $[-0.15,1.15]$ (§2.5.1).

GeomSubset
: Subconjunto de caras de un mesh USD; los de familia `materialBind` parten un mesh en piezas (§3.2).

GGX
: Distribución de microfacetas de Trowbridge–Reitz con parámetro $\alpha=r^2$; la de todos los lobes especulares (§4.1).

gpe
: Biblioteca de imágenes en GPU que adopta el device de slang-rhi; permite pasar meshes al efecto AOFX sin copias (§2.2, §3.1).

Guard (merge guard)
: Regla que hace infinito el coste de fusionar dos nodos: pieza, vidrio, espejo, fold, brillo, borde de color (§6.2.3).

Hamming distance
: Número de bits distintos entre dos máscaras; métrica del k-means de open directions y del codebook de oclusión (§6.7, §7.3).

Hard edge
: Arista abierta, no variedad o fold de más de 40°; limita el alcance de las Gaussians vecinas (§3.5).

Harder kernel
: Núcleo de Spark para opacidad $D>1$: $1-(1-g)^A$, $A=\exp((D^2-1)/e)$, la composición de $A$ Gaussians coincidentes (§2.5.2).

HDRI
: Imagen de entorno de alto rango dinámico usada como cielo; la del Corvette es `autoshop_01_4k` (§1.6, §4.10.4).

Hydra
: La arquitectura de render de OpenUSD: scene indices, render delegates y sus prims (§2.13).

IES profile
: Perfil fotométrico IESNA LM-63 que modula la radiancia de una luz, nunca su densidad (§2.10).

Importance sampling
: Muestrear direcciones con densidad parecida al integrando para reducir la varianza (§2.9, §2.10).

Indirect half
: Parte del transfer de los caminos que escapan tras rebotar: rgb, del color de lo que encontraron (§4.4.1).

Inline ray tracing
: Consultas de rayos (`RayQuery`) dentro de un kernel de cómputo, sin pipeline de ray tracing; la ruta de Metal (§2.3, §2.6).

Instancer
: Prim que dibuja copias de un prototipo; `PointInstancer` en USD, compuesto en el device como Storm (§2.8, §3.2).

Integrator
: Código que acumula la contribución de las partículas a lo largo de un rayo; las dos rutas del ray tracer comparten `rt_integrate.slang` (§2.6).

Irradiance
: Potencia incidente por unidad de área; la de un cuerpo sin transfer sale de nueve SH convolucionados (§4.3).

Jacobian
: Matriz de derivadas de una aplicación: de la perspectiva (§2.5.1), de la proyección inversa de un triángulo (§3.4) o del mapa de skinning con los gradientes de los pesos (§5.4.2).

Jitter
: Desplazamiento aleatorio dentro de un estrato o de un píxel; Halton (2, 3) entre pasadas del path tracer (§2.11).

k-means
: Agrupamiento que alterna asignar al centroide más cercano y recalcular centroides; ajusta la base CPCA y el diccionario de sombras (§7.2, §7.3).

KL divergence
: Divergencia de Kullback–Leibler entre distribuciones; la cota de Runnalls de su aumento por un merge es el coste de forma (§6.2.2).

Lattice
: La rejilla de cells sobre la que se emiten las Gaussians; sus footprints suman $2\pi\sigma^2$ (§3.3, §3.4).

LBVH
: Linear BVH de Karras construida en paralelo sobre códigos de Morton; ruta ComputeBvh y visibilidad bvh (§2.6, §2.8).

Leaf
: Una Gaussian con su bake, la hoja de un merge tree (§1.6).

Light BVH
: BVH sobre las luces con caja, cono de orientación y potencia por nodo, para elegir una luz por muestra (§2.10).

Light catcher
: Parche de Gaussians en el suelo con bake de la luz que arrojan las lámparas, dibujado de forma aditiva (§3.9).

Linear blend skinning
: Skinning que mezcla las transformaciones de los joints por pesos, $\mu'=M\sum_j w_jX_j(G\mu)$ (§5.4.1).

Lobe
: Un término de una BSDF con `eval`, `sample` y `pdf` (difuso, dieléctrico, conductor, sheen…); también un lobe zonal del transfer (§2.9, §4.4.5).

LobeStack
: Los lobes de un material, muestreados con MIS de una muestra por sus albedos (§2.9).

LOD
: Level of detail: dibujar menos nodos donde la vista no resuelve más; aquí, merge trees con cut por error (§6).

Mahalanobis radius
: Distancia normalizada por la covarianza; un splat se corta en $r^2_{\max}=\min(2\ln(255\alpha),9)$ (§2.5.1).

Master
: El fichero USDC de salida del bake, a precisión completa y juzgado frente al path trace, del que derivan los formatos (Parte I).

MaterialX
: Estándar de grafos de materiales; athenea lo compila a Slang (§2.9).

MCP
: Model Context Protocol; `athenea-mcp` sirve el motor por JSON-RPC 2.0 sobre stdio (§2.14).

Merge
: Un nodo interno de un merge tree, fusión de dos nodos por momentos (§1.6, §6.2).

Merge tree
: Árbol binario de merges por pares por pieza cuyo cut por error da el LOD (§6.2).

mesh2splat
: El algoritmo de EA que convierte un mesh en Gaussians por texel cubierto, y el comando y bundle de athenea que lo extienden (§3).

Microfacet model
: Modelo de superficie rugosa como microespejos con una distribución de normales (GGX), sombreado y Fresnel (§2.9).

Mip chain
: Cadena de niveles de una textura, cada uno la mitad; generada por un kernel que conserva la media (§2.3).

MIS
: Multiple importance sampling: combinar NEE y muestreo del material con la heurística de potencia (§2.11).

Moments
: Sumas $W$, $W\mu$, $W\Sigma$ ponderadas; los nodos de LOD son vectores de momentos que se suman (§6.1, §6.2.1).

Morton code
: Código que entrelaza los bits de las coordenadas; ordena splats y define cells de octree y candidatos de merge (§6.1, §6.2.4).

Motion blur
: Desenfoque por movimiento durante el shutter; en el raster, una actualización de rango uno de la covarianza (§2.5.1, §5.6).

NEE
: Next-event estimation: muestrear una luz directamente en cada vértice del camino (§2.11).

Occlusion mask
: Máscara conservadora por nodo de las direcciones desde las que alguna leaf se ve; con el ojo fuera del asset, un nodo cerrado hacia el ojo se descarta (§6.7).

Octahedral mapping
: Parametrización de la esfera sobre un cuadrado por un octaedro desplegado; la usan normales, open directions, el prefiltered map y la visibilidad por parte (§4.2, §4.3, §5.8).

Octree
: Árbol que divide el espacio en ocho; el LOD original agrupa por prefijos de Morton (§6.1).

OIDN
: Intel Open Image Denoise, el denoiser del path tracer (§2.11).

Opacity compensation
: Escalar la opacidad para conservar la energía al ensanchar un splat, y devolver el área perdida en el corte (`paidOpacity`, §2.5.1).

Open directions
: Los bits `shadowBits`: por cell de un grid octaédrico, si el rayo del bake salió de la escena; 256 bits en TX (§4.2, §4.4.2).

OpenColorIO
: Biblioteca de gestión de color usada como compilador de shaders (§2.12).

OpenFX
: API de plugins de compositores (Nuke, Natron); el nodo `Athenea 3DGS` dibuja una nube como generador (§2.14).

OpenPBR
: Uber-shader por capas de la ASWF; la pintura del Corvette es OpenPBR (§3.2, §3.6).

Oracle
: Implementación de referencia contra la que se comprueba otra; athenea no tiene CPU oracle y compara con kernels y formas cerradas (§2.1, §2.15).

Page
: Unidad de carga de un merge tree, nodos consecutivos en orden de banda; P = 3000 (§6.8, §7.5.1).

ParticleField
: Esquema USD de nubes de partículas; `ParticleField3DGaussianSplat` es el prim del master (§2.4, Parte I).

Path tracer
: Integrador Monte Carlo que sigue caminos de luz con rebotes; la verdad de referencia y el integrador del bake (§2.11).

PCG
: Familia de generadores aleatorios de O'Neill; el muestreador del path tracer pliega sus índices por un hash PCG (§2.11).

Piece
: Los triángulos de un mesh enlazados a un mismo material, la unidad de conversión y de merge (§3.2, §6.2).

Polish
: La reflexión especular de la base, distinta de la del coat (§4.6.3).

Pool
: Memoria de tamaño fijo del player donde residen pages y tramos, con desalojo desde la banda más fina (§7.6).

Power heuristic
: Peso del MIS $w=p_l^2/(p_l^2+p_m^2)$ (§2.11).

Prefiltered map
: El cielo sin su sol convolucionado con lobes GGX en una cadena octaédrica de ocho niveles (§4.3).

Prefix sum
: Suma acumulada paralela que da desplazamientos de escritura; una de las dos primitivas del motor (§2.3).

Primvar
: Atributo USD interpolado sobre un prim; los parámetros de athenea van como `primvars:athenea:splat:*` (Parte I).

PRT
: Precomputed radiance transfer: guardar por punto la proyección en SH de la visibilidad por el coseno (§1.5, §4.4.1).

PSNR
: Peak signal-to-noise ratio en dB sobre imágenes codificadas para pantalla (§1.6).

Radiance bake
: Guardar por Gaussian la luz que llegó bajo la iluminación del stage; exacta solo para ella (§4).

Radix sort
: Ordenación por dígitos; la por bloques ordena 10 M de pares en 12.4 ms (§2.3).

Range request
: Petición HTTP de un tramo de bytes; un par (bloque, tier) del `.athc` v3 es una (§7.1, §7.6).

Refit
: Recalcular las cajas de una BVH sin cambiar su topología cuando una pose mueve las partículas (§5.6).

Reflected field
: 16 coeficientes SH rgb por Gaussian de la radiancia rebotada por dirección bajo un cielo blanco, normalizados sobre las direcciones cerradas (§4.4.3).

Relight
: Iluminar de nuevo un asset bajo otro cielo y otras luces al dibujar, a partir de su transfer (§1.1, §4).

relMSE
: Error cuadrático medio relativo sobre valores lineales, $\overline{(a-b)^2/(b^2+10^{-2})}$ en el motor (§1.6, §2.15).

Render delegate
: El backend de Hydra que recibe y dibuja los prims; `hdAthenea` (§2.13).

Residency
: Lo que debe estar en memoria para dibujar un cut; las pages contienen unas 4× el cut (§6.8).

RGB9E5
: Formato de color de 32 bits con tres mantisas de 9 bits y un exponente compartido de 5; la emisión y el color dibujado del player (§2.4, §7.7).

Rig
: El skeleton con sus pesos y transformaciones, que el sistema lleva en vez de los frames (§5.1).

Roughness
: Rugosidad perceptual $r$ de un material, con $\alpha=r^2$ en GGX (§4.1).

Scene index
: Filtro de Hydra que transforma la escena antes del delegado; `HdAtheneaSkelSplatSceneIndex` (§2.13, §5.5).

Schlick (generalized)
: Fresnel aproximado entre un color de frente y uno rasante; el metal de OpenPBR y glTF con tinte F82 (§3.6, §4.6.4).

Shadow catcher
: Parche de Gaussians negras en el suelo cuya opacidad reproduce la sombra del objeto, $-\ln(\text{razón})/2\pi$ (§3.9, §7.9).

Shape operator
: Matriz simétrica $S$ con $d\mathbf n=S\,d\mathbf x$, la curvatura de la superficie bajo una Gaussian (§3.7, §4.8).

Sharp lobe
: Reflexión lisa que se evalúa por píxel en la mezcla, girando el espejo con el shape operator (§4.8.2).

Sheen
: Lobe de brillo de tejidos y microfibras, en sus formas de Imageworks y de Zeltner (§2.9, §4.6.4).

Shutter
: Intervalo en que el obturador está abierto; un splat arrastrado se convoluciona con un segmento (§2.5.1).

Skeleton / joint
: La jerarquía de articulaciones de UsdSkel y cada articulación; el gorrión tiene 609 joints (§5).

Skinning
: Deformar geometría por las transformaciones de los joints; aquí, de cada Gaussian con su Jacobian completo (§5.4).

Slang
: El lenguaje de shaders de todos los kernels, compilado a Metal, Vulkan, CUDA y WGSL (§2.1).

slang-rhi
: Capa de hardware de render de Slang sobre la que corre el motor (§2.1).

Spherical harmonics
: Base ortonormal de funciones en la esfera, índice $k=l^2+l+m$; grado 2 tiene 9 coeficientes y grado 3, 16 (§4.1).

Spill
: Fracción de lo que pinta un merge que sus leaves no pintan; evaluada y no adoptada (§6.5).

Splat
: Una Gaussian 3D dibujada por splatting; se usa como sinónimo de Gaussian (§1.6).

Split sum
: Aproximación del reflejo de entorno como producto del entorno prefiltrado por el albedo de la BRDF (§4.6.3).

SSIM
: Structural similarity index, medida de similitud perceptual (§3.5).

Stage
: Una escena USD compuesta; la entrada del bake tal como la entrega producción (Parte I, §3.2).

Still frame
: Política del player que, con la cámara quieta 300 ms, baja $\tau$ a 0.5 px y dibuja un cut fino (§7.10).

Storm
: El rasterizer de referencia de Hydra, usado como oracle de la visibilidad de meshes (§2.8).

Streaming
: Cargar datos por partes según la vista: chunks (§6.1), pages y tramos (§7.6).

Surfel
: Gaussian con una escala exactamente 0, un disco dibujado por intersección rayo–plano (§2.5.2, §3.4).

Thin lens
: Modelo de lente con desenfoque por el círculo de confusión, $R=fA|1/z-1/z_f|$ (§2.5.1, §2.11).

Thin wall
: Vidrio de lámina cuyas dos caras no desvían; cubre $2R/(1+R)$ y escala su reflejo (§3.8, §4.7).

Throughput
: Producto de los pesos de un camino, $f|\cos\theta|/p$ por vértice (§2.11).

Tier
: Nivel de un `.athc` v3 que agrupa secciones para que una petición traiga lo que necesita (§7.1).

Tile
: Bloque de 16 × 16 píxeles del rasterizer (§2.5).

Toksvig widening
: Ensanchar la roughness por la dispersión de normales bajo la Gaussian, $\alpha'^2=\alpha^2+2s^2$ (§3.7, §4.9).

Transfer
: El transporte de luz del bake: cómo el entorno de una Gaussian le llevaría cualquier cielo (§1.6, §4).

Transmittance
: Fracción de luz que pasa, $T=\prod(1-\alpha_j)$; el píxel se detiene bajo $10^{-4}$ (§2.5.2).

Traversal record
: Registro cuantizado de 5–11 bytes que el player recorre para elegir el cut (§7.5.3).

TX
: La forma actual del transfer: SH de grado 3 directa e indirecta, reflected field y 256 bits de open directions (§1.6, §4.2).

UDIM
: Convención de tiles de textura por rangos de UV, resuelta por una tabla de indirección (§2.9).

Unified memory
: Memoria compartida por CPU y GPU en Apple silicon; varios procesos pueden agotarla juntos (§2.3).

Unwidening
: Deshacer el ensanche de Toksvig al leer un sharp lobe, $r=\max(r'^4-2s^2,0)^{1/4}$ (§4.9.2).

UsdSkel
: El esquema de skeletons y skinning de USD (§5.2, §5.5).

Variant set
: Conjunto de alternativas de un prim USD; los clips del gorrión se eligen así (§5.5).

Vector quantization
: Sustituir un vector por el índice de su centroide; los bits de open directions (§7.3).

View transform
: Transformación de escena lineal a pantalla: Standard, AgX o ACES 2.0 (§2.12).

Visibility buffer
: Buffer de (instancia + 1, triángulo) por píxel a partir del cual se reconstruye el impacto (§2.8).

Visibility field (per part)
: Tabla de transmitancia direccional por parte del cuerpo, con bake en la bind pose y leída como producto (§5.8).

VNDF
: Distribución de normales visibles, la que muestrean los lobes de microfacetas (§2.9).

von Mises–Fisher
: Distribución de direcciones en la esfera; el GGX reflejado se toma como una de concentración $1/2\alpha^2$ al leer el campo (§4.6.2).

WASM
: WebAssembly; la traversal del player corre en un worker WASM (§7.5, §7.6).

WebGPU / WGSL
: La API de GPU de la web y su lenguaje de shaders, base del player (§7).

White furnace
: Prueba con un entorno uniforme blanco en la que un material sin pérdidas debe devolver 1 (§2.9, §2.11).

Workgroup memory
: Memoria compartida por un grupo de hilos; la mezcla carga allí los registros de su tile (§2.5.2).

Zonal harmonics
: SH simétricos alrededor de un eje, que giran girando el eje; dos lobes por Gaussian con skinning (§4.4.5, §5.7).
