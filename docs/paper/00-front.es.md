<!-- Copyright (c) 2026 jesus luque. -->

::: {#front}

# athenea: un baker de relight para Gaussian splatting a partir de assets USD de producción {.unnumbered .unlisted .paper-title}

<p class="byline">jesus luque</p>

<p class="date">19 de octubre de 2026</p>

## Resumen {.unnumbered .unlisted}

Se describe athenea, un baker de relight para Gaussian splatting 3D: convierte meshes USD de producción y sus
materiales en Gaussians y hace el bake, en cada Gaussian, del transporte de luz necesario para su relight al dibujar,
siempre con un path trace del mesh de origen como referencia. Un mesh se convierte en el device en Gaussians
anisótropas, una por cell de una lattice derivada de una cámara, con opacidad que respeta la coverage, bordes sujetos
donde una superficie termina o forma un fold, materiales OpenPBR por capas, vidrio clasificado como sólido o como pared
fina, y roughness ensanchada por la dispersión de normales bajo cada Gaussian. El propio path tracer de GPU del
renderer hace después el bake de un transfer en cada Gaussian: spherical harmonics (SH) de grado 3 de la luz
directa e indirecta, un reflected field, 256 bits de open directions y las marcas del material. Al dibujar, estos
datos hacen el relight de la nube bajo cualquier cielo y cualquier luz, con reflejos nítidos girados por píxel mediante
un shape operator almacenado. En un Chevrolet Corvette C7 de 192 meshes, doblar la resolución del bake de 640 a
1280 px subió el PSNR del coche con relight frente al path trace de 16.55 a 17.88 dB en su recorte y de 22.18 a
23.90 dB en el frame, y para las vistas de cerca se hizo el bake de un close-up level a 5120 px de 74.5 millones de
leaves en 24 trees. El baker se apoya en un renderer de USD e Hydra en el que cada paso que toca datos de la escena
se ejecuta como un kernel de cómputo en la GPU. Los assets con rig conservan su esqueleto, con un Jacobian que incluye
los gradientes de los pesos, lobes zonales de transfer que giran con un miembro y campos de visibilidad por parte.
Merge trees por pieza con un error de pantalla monótono y acotado, culls de caras traseras y de oclusión, y pages
ordenadas por error dan el LOD con un coste de unos 0.1 dB. Un transfer se lee en el sistema de referencia de su propia
nube, con el cielo girado a él una vez por instancia, así que una nube que la escena gira hace el relight igual que sin
girar (relMSE $9\times10^{-6}$). Cada parámetro que lleva una Gaussian, desde su albedo y su normal hasta la visibilidad
del cielo, el bounce y el reflected field del transfer, se puede dibujar a través de una cámara como un canal de imagen
propio. La salida del bake es un master: un fichero USD con cada parámetro a precisión completa, juzgado frente al path
trace, que leen tal cual el motor, un nodo OpenFX en un compositor (con capas de profundidad, posición en el mundo y
Cryptomatte y un cielo que elige el artista) y un add-on de Blender. Un proceso aparte, con su propio objetivo, deriva
del master un formato web para un móvil: un PCA por clusters del transfer, la visibilidad cuantizada vectorialmente y un
registro compacto de 40 bytes llevan el coche con relight a un player WebGPU en un iPhone, con su pérdida medida frente
al master.

**Palabras clave:** relight baking; Gaussian splatting 3D; precomputed radiance transfer; spherical harmonics; USD;
Hydra; de mesh a Gaussians; path tracing; LOD; PCA por clusters; cadenas solo en GPU; WebGPU; OpenFX; composición;
canales de imagen.

## Índice {.unnumbered .unlisted}

::: {#contents}

| 1. Introducción
| 2. El motor y el renderer athenea
| 3. De meshes USD a Gaussians (mesh2splat)
| 4. Transporte de luz en el bake de las Gaussians (transfer) y relight
| 5. Assets animados: skeletons, skinning y visibilidad por partes (el peón y el gorrión)
| 6. LOD: merge trees, cuts por error, pages y oclusión
| 7. Aplicación: compresión y un player WebGPU en móvil
| 8. Evaluación
| 9. Limitaciones y trabajo futuro
| 10. Referencias
| Apéndice A. Repositorios y comandos

:::

:::

::: {.nota}
**Notas al título y al resumen.** Las notas que siguen comentan, en el orden en que aparecen, los términos del título y del resumen. Se colocan aquí, tras el cierre del bloque `#front`, para no alterar la composición de la portada ni el índice.

**Baker, bake.** En producción, *bake* («hornear») es precalcular una magnitud cara —iluminación, oclusión, normales, desplazamiento— y guardarla en un soporte barato de leer al dibujar: texturas (*lightmaps*, *AO maps*), vértices o, aquí, los propios Gaussians. Un *baker* es el programa que hace ese precálculo. La idea clave es mover el coste desde el momento de dibujar (cada frame, en tiempo real) al momento de preparar el asset (una vez, sin prisa), a cambio de memoria y de pérdida de generalidad: lo que se ha precalculado ya no cambia si la escena cambia. El matiz de athenea es que no se hornea la *luz* (la radiancia final), sino el *transporte* de la luz, que es independiente del cielo; por eso el resultado admite relight.

**Relight.** Volver a iluminar un objeto bajo una iluminación distinta de la que tenía cuando se capturó o se precalculó. Matemáticamente, si la radiancia saliente es lineal en la radiancia incidente $L_{\text{in}}$, existe un operador lineal $\mathcal{T}$ (el transporte) tal que $L_{\text{out}} = \mathcal{T}\,L_{\text{in}}$. Guardar una aproximación de $\mathcal{T}$ por punto permite aplicarlo a cualquier $L_{\text{in}}$ nuevo; guardar solo $L_{\text{out}}$ (como hace un 3DGS entrenado con fotografías) congela una única iluminación.

**USD de producción.** OpenUSD (Universal Scene Description, Pixar, [43] del §10) es el formato y el sistema de composición de escenas que usan los estudios: un *stage* se compone de capas (*layers*) mediante *references*, *payloads*, *variants* y *overrides*; la geometría son *prims* `UsdGeomMesh`, los materiales *networks* `UsdShade` (aquí MaterialX/OpenPBR), las luces `UsdLux`, los esqueletos `UsdSkel`. «De producción» quiere decir que el asset no está preparado para el baker: tiene cientos de meshes, materiales por capas, texturas UDIM, instancias y jerarquías de transformaciones tal como salen de un DCC.

*Referencias:* [43] del §10; especificación en openusd.org («Introduction to USD», «Glossary»); código: `modules/usd/src/MeshStage.cpp` (lectura del stage para la conversión), `modules/usd/src/Export.cpp` (escritura del master).
:::

::: {.nota}
**Gaussian splatting 3D (3DGS).** Representación de una escena como un conjunto de Gaussianas 3D anisótropas que se proyectan a pantalla y se mezclan por orden de profundidad (Kerbl et al. 2023, [1]). Cada Gaussian $i$ es una función de densidad no normalizada
$$G_i(\mathbf{x}) = \exp\!\left(-\tfrac12 (\mathbf{x}-\boldsymbol{\mu}_i)^{\mathsf T}\,\Sigma_i^{-1}\,(\mathbf{x}-\boldsymbol{\mu}_i)\right),$$
con centro $\boldsymbol{\mu}_i$, covarianza $\Sigma_i$, una opacidad $o_i\in[0,1]$ y un color, que en 3DGS es una función de la dirección de vista expresada en spherical harmonics.

**Anisótropa, covarianza.** «Anisótropa» significa que la Gaussian no es una esfera: sus elipsoides de nivel tienen tres ejes de longitudes distintas. La covarianza se parametriza para que siempre sea simétrica y definida positiva: $\Sigma = R\,S\,S^{\mathsf T}R^{\mathsf T}$, con $R$ una rotación (guardada como cuaternión) y $S=\mathrm{diag}(s_x,s_y,s_z)$ las escalas (desviaciones típicas a lo largo de los ejes propios). Si una escala es casi cero, la Gaussian degenera en un disco: es lo que athenea llama *surfel* (§2.5.2), y es la forma natural de una Gaussian que representa un trozo de superficie de un mesh.

**Splat, splatting.** *Splatting* es la técnica de render por «salpicadura»: en lugar de lanzar rayos, cada primitiva se proyecta a la pantalla y deposita su huella (*footprint*) en los píxeles que cubre. Un *splat* es esa huella y, por extensión, la primitiva. En la proyección EWA (Zwicker et al. [3]) la Gaussian 3D se transforma en una Gaussian 2D en pantalla con covarianza
$$\Sigma' = J\,W\,\Sigma\,W^{\mathsf T}J^{\mathsf T},$$
donde $W$ es la parte lineal de la transformación mundo→cámara y $J$ el Jacobiano de la proyección perspectiva evaluado en el centro (la aproximación afín local). En cada píxel $\mathbf{p}$ la Gaussian aporta $\alpha_i = o_i \exp(-\tfrac12 \mathbf{d}^{\mathsf T}\Sigma'^{-1}\mathbf{d})$, $\mathbf{d}=\mathbf{p}-\boldsymbol{\mu}'_i$, y el color se compone de delante hacia atrás:
$$C = \sum_i c_i\,\alpha_i \prod_{j<i}(1-\alpha_j).$$

**Por qué 3DGS es atractivo aquí.** La representación es explícita (no hay red neuronal que evaluar), la rasterización por tiles es muy paralela y de coste predecible, y una Gaussian puede representar con una sola primitiva detalle subpíxel, transparencia parcial y bordes suaves. El precio: el orden por profundidad es por Gaussian (no por píxel), lo que produce *popping* cuando dos Gaussians cruzan su orden, y la memoria por primitiva es grande (decenas a cientos de bytes con SH).

*Referencias:* [1], [3] del §10; código: `shaders/athenea/common/packing.slang` (`encodeQuaternion`, `axesOfQuaternion`, `isSurfel`), `shaders/athenea/splat/splat_project.slang` (proyección EWA, `splatProject`), `shaders/athenea/splat/splat_blend.slang` (`blendPixel`, composición por tiles), `modules/scene/include/athenea/scene/GpuClouds.h` (`GpuSplats`, la disposición común de la nube); `docs/decisions.md`, sección «Ray tracing Gaussians».
:::

::: {.nota}
**Path trace como referencia.** *Path tracing* (Kajiya, «The Rendering Equation», SIGGRAPH 1986) resuelve la ecuación de render
$$L_o(\mathbf{x},\omega_o) = L_e(\mathbf{x},\omega_o) + \int_{\Omega} f_r(\mathbf{x},\omega_i,\omega_o)\,L_i(\mathbf{x},\omega_i)\,(\mathbf{n}\cdot\omega_i)\,d\omega_i$$
por Monte Carlo: se trazan caminos aleatorios desde la cámara, rebotando en las superficies según su BSDF $f_r$, y se promedia. Con $N$ caminos el estimador es insesgado y su error típico decrece como $1/\sqrt{N}$. Es la referencia porque, con suficientes muestras, converge a la solución física del modelo de escena (geometría, materiales, luces) sin aproximaciones de shading. En el artículo, «fiel al path trace» significa que la imagen de las Gaussians con relight se compara píxel a píxel con la imagen que el mismo motor produce trazando el mesh original bajo la misma iluminación.

**El mesh de origen.** Se compara con el mesh, no con una nube: así se mide todo el error introducido por el pipeline (conversión + bake + relight + LOD), no solo el de un paso.

*Referencias:* [21] del §10 (MIS); Kajiya (1986); Pharr, Jakob y Humphreys, *Physically Based Rendering*, 4.ª ed. (2023), caps. 2 y 13; código: `modules/technique/src/PathTracer.cpp` (`PathTracer::trace`, `PathTracer::bake`); `docs/decisions.md`, sección «Complete USD: the path tracer (M6)».
:::

::: {.nota}
**Conversión en el device.** *Device* es el término de las APIs de GPU (Metal `MTLDevice`, Vulkan `VkDevice`, CUDA context) para la GPU y su memoria, frente al *host* (la CPU). «En el device» significa que los datos se suben una vez y todo el cálculo se hace con kernels de cómputo sobre buffers de GPU.

**Una Gaussian por cell de una lattice derivada de una cámara.** Cada triángulo se recorre sobre una rejilla regular (*lattice*) en el plano de dos de sus ejes, y cada celda (*cell*) cubierta emite una Gaussian. El tamaño de la cell se deriva de la cámara que va a mirar: la cell de una pieza es lo que cubre un píxel donde la pieza está más cerca de la cámara,
$$\text{cell} = z\cdot\frac{\text{aperture}/\text{pixels}}{\text{focal}},$$
con $z$ la distancia de la cámara a la caja de la pieza. Es la regla de Mesh2GS [13]: una Gaussian por píxel proyectado hace la densidad «justa» para todas las piezas.

**Opacidad que respeta la coverage.** Si la superficie cubre una fracción $F$ (1 para una superficie opaca, menos con un mapa de opacidad o en un vidrio), la opacidad $\alpha$ de cada Gaussian no puede ser simplemente $F$: Gaussians vecinas se solapan, y la transmitancia acumulada es $T=\prod(1-\alpha g)$. Tomando $-\ln T \approx \sum \alpha g + \tfrac12\alpha^2 g^2$ sobre una lattice, con $\sum g = 2\pi\sigma^2$ y $\sum g^2 = \pi\sigma^2$ por cell (en unidades de cell), la ecuación a resolver es
$$\tfrac{\pi\sigma^2}{2}\,\alpha^2 + 2\pi\sigma^2\,\alpha = -\ln(1-F),$$
cuya raíz positiva se usa, más $1/255$ por la cola de la Gaussian que no se dibuja.

**Bordes sujetos (edges held).** Una Gaussian tiene la anchura de una cell; donde una superficie termina (borde abierto) o se dobla (pliegue o *fold*, una arista con ángulo diedro grande) una Gaussian ancha sobresale. La conversión detecta esas aristas en el device y reduce la varianza de la Gaussian en la dirección perpendicular a la línea, hasta su distancia a ella.

*Referencias:* [12], [13] del §10; código: `shaders/athenea/usd/mesh2splat_cells.slang` (la cell desde una cámara), `plugins/mesh2splat/mesh2splat.slang` (`m2sCoverageAlpha`, `m2sHoldEdges`, `projectionAxes`), `shaders/athenea/usd/mesh2splat_edges.slang`; `docs/decisions.md`, secciones «A converted surface covers what its opacity says» y «mesh2splat: a mesh becomes a cloud, through the plugin interface».
:::

::: {.nota}
**OpenPBR por capas.** OpenPBR Surface ([30]) es el modelo de material «uber-shader» estándar de la Academy Software Foundation, sucesor de Autodesk Standard Surface y Adobe Standard Material. Describe una superficie como una pila de capas físicas: *fuzz* (sheen), *coat* (barniz dieléctrico con su propio IOR y rugosidad), *specular* sobre una *base* que puede ser dieléctrica difusa, metálica, transmisiva o subsurface, con *thin-film*. Cada capa atenúa lo que deja pasar a la de abajo según su Fresnel. Una pintura de coche es típicamente base metálica/flakes + coat; athenea captura por Gaussian el peso y los parámetros de cada lobe (specular, coat, sheen) para evaluarlos al dibujar.

**Vidrio sólido frente a pared fina.** Un dieléctrico *sólido* (una bola, un faro macizo) refracta: el rayo entra, se curva por Snell, recorre el volumen y sale por otra cara. Una *pared fina* (*thin-walled*, un parabrisas modelado como una sola superficie, o dos caras paralelas a 3–4 mm) no desvía apreciablemente la imagen: lo que no se refleja pasa recto. Para una lámina con reflectancia de Fresnel normal $R=((\eta-1)/(\eta+1))^2$ en cada cara, sumando las reflexiones múltiples, la fracción reflejada es $2R/(1+R)$; para $\eta=1.5$, $R=0.04$ y la lámina refleja $\approx 7.7\,\%$. athenea clasifica automáticamente: una pieza es lámina si tiene aristas abiertas o si su espesor medio $2V/A$ (volumen sobre área) es pequeño frente a la cell.

**Roughness ensanchada por la dispersión de normales (Toksvig).** Una Gaussian tiene una sola normal, pero bajo ella la superficie original puede curvarse; un reflejo nítido que en el mesh se mueve a lo largo de una moldura cromada aparece en la Gaussian como una mancha de la anchura de la Gaussian. Toksvig (2005, [20]) propuso tratar la variación de normales bajo un texel como varianza adicional de la distribución de microfacetas. En athenea, con $\alpha=r^2$ ($r$ la roughness perceptual) y $\sigma_n$ la dispersión angular de normales bajo la Gaussian,
$$\alpha'^2 = \alpha^2 + 2\sigma_n^2 .$$

*Referencias:* [20], [30] del §10; especificación OpenPBR en academysoftwarefoundation.github.io/OpenPBR; código: `plugins/mesh2splat/mesh2splat.slang` (`m2sSpread`, `m2sWidened`, `m2sGlassCovers`), `apps/athenea/src/CmdMesh2Splat.cpp` (`classifySheets`, `thinGlassOf`), `shaders/athenea/usd/mesh2splat_sheet.slang`; `docs/decisions.md`, secciones «A reflection off a curved strip is the mean over the gaussian (Toksvig)» y «A windscreen modelled as one surface is a sheet (`--thin-glass`)».
:::

::: {.nota}
**Spherical harmonics (SH).** Las armónicas esféricas reales $Y_{lm}(\omega)$, $l\ge 0$, $-l\le m\le l$, son una base ortonormal de las funciones sobre la esfera, análoga a la serie de Fourier sobre el círculo: $l$ es la «frecuencia» angular (*band*). Truncar en grado $L$ da $(L+1)^2$ funciones: 1, 4, 9 o 16 para grados 0 a 3. Una función $f(\omega)$ se aproxima por $f \approx \sum_{l\le L,m} f_{lm} Y_{lm}$ con $f_{lm} = \int f\,Y_{lm}\,d\omega$. La propiedad que hace útil a SH para iluminación: el producto escalar de dos funciones se convierte en el producto escalar de sus vectores de coeficientes, $\int f g\,d\omega \approx \sum_k f_k g_k$. Grado 3 representa variaciones de unos $180°/3 \approx 60°$: suficiente para luz difusa y sombras suaves, no para reflejos nítidos.

**Transfer de la luz directa e indirecta (PRT).** Es el *precomputed radiance transfer* de Sloan, Kautz y Snyder ([14]). Para una superficie difusa con albedo $\rho$ iluminada por un entorno lejano $L(\omega)$,
$$L_o = \frac{\rho}{\pi}\int_{\Omega} L(\omega)\,V(\omega)\,\max(\mathbf{n}\cdot\omega,0)\,d\omega \approx \frac{\rho}{\pi}\sum_k L_k\,T_k,$$
donde $V$ es la visibilidad y $T_k = \int V(\omega)\max(\mathbf{n}\cdot\omega,0)\,Y_k(\omega)\,d\omega$ es el *transfer*: no depende del cielo, se precalcula una vez por punto. La parte *directa* es esta (lo que llega del cielo sin rebotar, un número por coeficiente); la *indirecta* añade lo que llega tras rebotar en otras partes del asset, con su color (tres números por coeficiente). En athenea, un transfer TX de grado 3 son $16 + 48 + 48 = 112$ valores f16 por Gaussian: 16 de la mitad directa (escalar), 48 de la indirecta (rgb) y 48 del reflected field (16 coeficientes rgb). El bake no proyecta: *ajusta* los coeficientes por mínimos cuadrados sobre el hemisferio que la superficie mira.

**Reflected field.** Lo que ve un reflejo en las direcciones *cerradas* (ocluidas por el propio asset): el suelo, la carrocería en el cromo, el interior oscuro tras un parabrisas. Se guarda normalizado sobre esas direcciones y se acopla al cielo al dibujar, de forma que cambia si cambia el cielo.

**256 bits de open directions.** Una rejilla octaédrica de $16\times16$ celdas sobre toda la esfera; cada bit dice si un rayo en esa dirección escapó al cielo. Una celda subtiende unos $11°$. Al dibujar, cada lobe especular se lee como un cono sobre estas celdas para saber qué fracción de su reflexión ve el cielo (y por tanto se toma del mapa de entorno) y qué fracción está cerrada (y se toma del reflected field).

**Marcas del material.** Bits y bytes por Gaussian con el tipo de lobes (pesos de specular, coat, sheen, IOR, metalness, roughness original antes de ensanchar) que el relight necesita para evaluar el material.

*Referencias:* [14], [17] del §10; Green, R., «Spherical Harmonic Lighting: The Gritty Details» (GDC 2003); código: `shaders/athenea/common/transfer_layout.slang` (recuentos 9/16/36/64/84/112), `shaders/athenea/common/sh.slang` (`shBasisValue`, `evaluateRest`), `shaders/athenea/common/bake_fit.slang` (`bakeFit`), `shaders/athenea/usd/mesh2splat_bake.slang` (`m2sTransferInto`), `modules/technique/include/athenea/technique/PathTracer.h` (`BakePoints`); `docs/decisions.md`, secciones «A transfer right under any sky and any light (task TX)» y «The bake fits the harmonics, and no longer projects them».
:::

::: {.nota}
**Relight al dibujar y reflejos nítidos girados por píxel.** Al dibujar, la parte difusa es un producto escalar del transfer con los 16 coeficientes del cielo. La parte especular no cabe en SH de grado 3; se evalúa con los lobes del material contra el cielo prefiltrado (una cadena de mips octaédrica por roughness, *split-sum*). Como una Gaussian tiene una sola normal, el reflejo sería constante sobre toda su huella; para reproducir cómo gira el reflejo sobre una superficie curva se guarda el **shape operator** (operador de forma, o endomorfismo de Weingarten) $S$, la matriz $2\times2$ simétrica que da la variación de la normal con la posición sobre la superficie: $d\mathbf{n} = S\,d\mathbf{x}$. Sus autovalores son las curvaturas principales $\kappa_1,\kappa_2$. En cada píxel se conoce el desplazamiento $d\mathbf{x}$ desde el centro de la Gaussian y se gira la normal por $S\,d\mathbf{x}$, de modo que el reflejo se mueve dentro de la Gaussian como lo haría en el mesh. Se guardan tres números $(S_{uu}, S_{uv}, S_{vv})$ en los ejes propios de la Gaussian.

*Referencias:* do Carmo, *Differential Geometry of Curves and Surfaces* (1976), §3.2; Karis, «Real Shading in Unreal Engine 4» (SIGGRAPH 2013, split-sum); código: `plugins/mesh2splat/mesh2splat.slang` (`m2sShape`), `shaders/athenea/splat/splat_relight.slang` (`specularLobe`, `splatFieldAlong`, `splatLobeOpen`), `shaders/athenea/splat/splat_project.slang` (el giro por píxel y el *fold*).
:::

::: {.nota}
**Chevrolet Corvette C7 de 192 meshes; PSNR.** El PSNR (*peak signal-to-noise ratio*) entre una imagen $I$ y la referencia $K$, con valores en $[0,1]$, es
$$\mathrm{PSNR} = -10\log_{10}\mathrm{MSE}, \qquad \mathrm{MSE} = \frac{1}{3N}\sum_{p,c}\big(I_{p,c}-K_{p,c}\big)^2 ,$$
en decibelios (dB). Es logarítmico: +1 dB equivale a multiplicar el MSE por $10^{-0.1}\approx 0.79$, +3 dB a dividirlo por 2, +10 dB a dividirlo por 10. En athenea se calcula sobre la imagen codificada para pantalla (sRGB), que es como la ve una persona.

**Cómo leer 16.55 → 17.88 dB y 22.18 → 23.90 dB.** En el recorte del coche, +1.33 dB es un MSE $10^{-0.133}\approx 0.74$ veces el anterior: un 26 % menos de error cuadrático al doblar la resolución del bake. En el frame entero, +1.72 dB es un MSE $\approx 0.67$ veces el anterior. El PSNR del frame es mayor que el del recorte porque el frame incluye mucho fondo (el cielo de entorno, idéntico en ambas imágenes) que diluye el error: por eso el recorte es la métrica exigente. Valores en torno a 17–24 dB son bajos en términos absolutos (un códec de vídeo suele dar 35–45 dB), pero aquí se compara una representación completamente distinta (Gaussians reiluminadas) con un path trace del mesh, con reflejos nítidos y vidrio, y diferencias de fase en reflejos pequeños penalizan mucho al MSE. Lo relevante es la mejora relativa entre pasos y los cocientes por clase (§1.6, §8).

**Close-up level a 5120 px, 74.5 millones de leaves en 24 trees.** Un bake cuya cell se deriva de una cámara de 5120 píxeles de ancho (cuatro veces 1280): cada pieza tiene Gaussians del tamaño de un píxel de esa cámara, para vistas cercanas. *Leaves* son las Gaussians con bake (las hojas de los merge trees) y *trees* los merge trees por grupos de piezas. Una medida intermedia del desarrollo dio 69.5 M leaves, con una parte y la oclusión aún pendientes; la cifra del artículo es la del nivel terminado.

*Referencias:* Wang, Bovik et al., «Image Quality Assessment: From Error Visibility to Structural Similarity» (2004) para SSIM y las limitaciones del PSNR; código: `plugins/measure/Measure.cpp` (el PSNR de la imagen codificada, `-10 log10(mse)`), `plugins/measure/measure.slang`, `apps/athenea/src/CmdCompare.cpp`; la sección 7 (el Corvette en un iPhone) y la sección 8 del artículo.
:::

::: {.nota}
**Renderer de USD e Hydra; cada paso como kernel de cómputo en la GPU.** *Hydra* es la arquitectura de render de OpenUSD: separa la escena (*scene delegate* / *scene index*, que lee USD) del renderer (*render delegate*, `HdRenderDelegate`), que recibe *rprims* (mesh, points, curves), *sprims* (cámaras, luces, materiales) y *bprims* (buffers) mediante llamadas `Sync` y dibuja en `HdRenderPass::Execute`. Un *render delegate* es un plugin: el mismo delegado `hdAthenea` puede cargarse desde `usdview`, desde Blender (que tiene su propio Hydra) o desde el motor. Un *kernel de cómputo* (*compute shader*) es un programa de GPU que se lanza sobre una rejilla de hilos sin pasar por el pipeline gráfico; aquí se escribe en Slang ([44]) y se compila a Metal, Vulkan o CUDA. La regla «cada paso que toca datos es un kernel» evita copias CPU↔GPU y garantiza que el baker escala con la GPU; la CPU solo lee ficheros y lleva cuentas.

*Referencias:* [43], [44] del §10; documentación de Hydra en openusd.org («Hydra 2.0», «Render Delegate»); código: `modules/usd/src/RenderDelegate.h` (`HdAtheneaRenderDelegate`), `modules/gpu/src/ComputeKernel.cpp` (`ComputeKernel::dispatch`, parámetros ligados por nombre); `CLAUDE.md` del repositorio, regla «No CPU arithmetic on data».
:::

::: {.nota}
**Assets con rig: esqueleto, Jacobian con gradientes de pesos.** Un *rig* es la estructura que deforma un modelo: un esqueleto de *joints* y, por vértice, pesos $w_j$ que dicen cuánto lo mueve cada joint (*linear blend skinning*, LBS): $\mathbf{p}' = \sum_j w_j X_j\,\mathbf{q}$. Para una Gaussian no basta con mover el centro: hay que transformar su covarianza, $\Sigma' = J\,\Sigma\,J^{\mathsf T}$, con $J$ el Jacobiano del mapa. Como los pesos varían con la posición, el Jacobiano completo es
$$J = L(M)\Big[\sum_j w_j A_j L(G) + \sum_j (X_j \mathbf{q})\,\nabla w_j^{\mathsf T}\Big],$$
y el segundo término —el que la mayoría de implementaciones omiten— estira la Gaussian donde dos joints discrepan (en el ala del gorrión, entre un 15 y un 50 % en los percentiles 90 y 99).

**Lobes zonales que giran con un miembro.** Un transfer en SH en coordenadas del mundo no gira con el hueso; girarlo cuesta una matriz $9\times9$ o $16\times16$ por Gaussian. Un *lobe zonal* es una función simétrica respecto de un eje $\mathbf{a}$, $g(\omega) = \sum_l z_l k_l Y_{l0}$ en el sistema del eje; girarlo es girar el eje. Se ajusta (Sloan 2008, [16]) el eje y los coeficientes $z_l$ que mejor aproximan el transfer, en el sistema propio de la Gaussian.

**Campos de visibilidad por parte.** Cada parte del cuerpo hornea, en su pose de bind, una rejilla de sondas con cuánta luz bloquea en cada dirección; al dibujar, la visibilidad de una Gaussian es el producto de lo que responde cada parte en su propio sistema actual. Sin rayos.

*Referencias:* [16], [41], [42] del §10; Magnenat-Thalmann et al. (1988) y Kavan et al. (2008) sobre LBS; código: `shaders/athenea/scene/splat_skin.slang` (`splatSkin`), `modules/scene/src/SplatSkinner.cpp`, `plugins/splattransferzonal/splattransferzonal.slang`, `shaders/athenea/splat/splat_visibility.slang`, `modules/technique/src/SplatVisibility.cpp`; `docs/decisions.md`, secciones «The whole Jacobian of a skinned gaussian», «A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame» y «A cloud shadows itself by part: baked once, read every frame, no ray».
:::

::: {.nota}
**LOD: merge trees, error de pantalla monótono y acotado, culls y pages.** *LOD* (*level of detail*) es dibujar menos primitivas cuando el objeto se ve pequeño. Un *merge tree* es un árbol binario cuyas hojas son las Gaussians originales y cuyos nodos internos son Gaussians que aproximan a sus dos hijos (merge por pares, igualando momentos: $w = w_a + w_b$, $\boldsymbol\mu = (w_a\boldsymbol\mu_a + w_b\boldsymbol\mu_b)/w$, $\Sigma = \sum_i \tfrac{w_i}{w}(\Sigma_i + (\boldsymbol\mu_i-\boldsymbol\mu)(\boldsymbol\mu_i-\boldsymbol\mu)^{\mathsf T})$). Cada nodo lleva un error $\varepsilon$ en metros, forzado a ser **monótono** hacia la raíz ($\varepsilon(n) \ge \varepsilon(\text{hijos})$) y una esfera envolvente $\rho(n)$; su error proyectado es $P(n) = \varepsilon\,\text{ppu}/\max(d-\rho,\text{near})$. El *cut* para una vista son los nodos con $P(n) \le \tau < P(\text{padre})$: la monotonía garantiza que basta una prueba local por nodo, en paralelo, sin recorrer ancestros (la idea de Nanite, [35]). *Cull de caras traseras*: un nodo cuyo cono de normales mira entero hacia atrás en una closed shell no se dibuja. *Cull de oclusión*: un nodo cuya máscara de direcciones abiertas está cerrada en la dirección de la cámara no puede verse desde fuera. *Pages*: bloques de nodos ordenados por error, para hacer streaming de lo grueso primero.

**«Con un coste de unos 0.1 dB».** La imagen dibujada con el cut pierde unos 0.1 dB de PSNR frente a dibujar todas las leaves: un MSE solo un 2.3 % mayor ($10^{0.01}$), es decir, el LOD casi no se nota en la métrica.

*Referencias:* [2], [9], [34], [35] del §10; código: `shaders/athenea/lod/lod_pairs.slang` (`mergeCost`, `pairMerge`), `shaders/athenea/lod/lod_tree_cut.slang` (el error monótono y la esfera), `shaders/athenea/lod/lod_pages.slang`, `shaders/athenea/lod/occlusion_mask.slang`, `modules/usd/src/Decimate.cpp`; `docs/decisions.md`, secciones «Simplify: fewer gaussians by pairs, and surfels» y «Decimation: the levels of detail, cut once by what a merge loses».
:::

::: {.nota}
**El transfer en el sistema de referencia de su nube; relMSE $9\times10^{-6}$.** Los coeficientes SH, el reflected field y las celdas abiertas son direcciones del fichero en el que se hizo el bake. Si la escena gira la nube (un asset Y-up en una escena Z-up), hay que expresar el cielo en el sistema de la nube: los coeficientes girados son $c'_k = \int Y_k(\omega)\,L(R\,\omega)\,d\omega$, que se calculan una vez por instancia con una cuadratura exacta para productos de grado 6 (cuatro nodos de Gauss–Legendre en $z$ por ocho pasos en $\phi$). El *relMSE* es el error cuadrático medio relativo sobre valores lineales, $\frac{1}{N}\sum (a-b)^2/(b^2+\epsilon)$ con $\epsilon = 10^{-2}$; $9\times10^{-6}$ equivale a un error relativo típico del 0.3 % (raíz cuadrada), y `docs/decisions.md` registra 61 dB de PSNR en esa misma prueba: el peón girado 60° es prácticamente idéntico al no girado. La prueba de control (girar en el sentido contrario) da 11 dB, lo que demuestra que la métrica distingue los sistemas.

*Referencias:* Gauss–Legendre: Abramowitz y Stegun, *Handbook of Mathematical Functions*, §25.4; código: `shaders/athenea/splat/sky_turn.slang` (`skyTurn`), `shaders/athenea/splat/splat_relight.slang` (`relightCloudFrame`, `transferAlong`), `shaders/athenea/reference/image_compare_hdr.slang` (relMSE); `docs/decisions.md`, sección «A transfer the stage turns, and the gaussians' parameters drawn (19 Oct)».
:::

::: {.nota}
**Cada parámetro como un canal de imagen (AOV).** Un *AOV* (*arbitrary output variable*) o canal de imagen es una imagen adicional que el renderer produce junto al color: profundidad, normales, albedo, IDs. Aquí cualquier parámetro de la Gaussian —albedo, roughness (ensanchada o la del material), metalness, transmisión, IOR, normal, opacidad, tamaño, visibilidad del cielo, bounce, fracción de celdas abiertas, emisión, reflected field, luz con bake— se compone a través de la cámara como si fuera su color. Sirve para depurar el bake, para la composición y como datos de entrenamiento para un relight aprendido.

*Referencias:* código: `shaders/athenea/splat/splat_relight.slang` (constantes `kSplatViewAlbedo` … `kSplatViewMaterialRoughness`), `modules/render/src/TileRasterizer.cpp` (parámetro `splatView`), `modules/mcp/src/Tools.cpp` (argumento `splatView` de la herramienta `render`); ajuste USD `athenea:splatView` en `modules/usd/src/StageRenderer.cpp`.
:::

::: {.nota}
**El master: USD a precisión completa; nodo OpenFX; Cryptomatte; add-on de Blender.** El *master* es el resultado de referencia del bake, en el sentido del «máster» de audio o vídeo: la copia de máxima calidad de la que se derivan las versiones de distribución. Es un USDC (USD binario *crate*) con la nube como `ParticleField` y sus atributos (transfer, celdas, shape operator, lobes) sin cuantizar. *OpenFX* (OFX) es el estándar abierto de plugins de efectos de imagen que usan Nuke, Natron, DaVinci Resolve o Flame; el nodo de athenea es un *generator* que dibuja la nube a la cámara del compositor, en un proceso aparte. Las capas que entrega: profundidad (z de vista), posición en el mundo (P) y *Cryptomatte* (Friedman y Jones, [47]): un canal de IDs por hash de nombre con cobertura por rangos, que permite al compositor sacar mates de cualquier objeto con antialiasing y transparencia correctos. El add-on de Blender registra `hdAthenea` como motor de render a través del Hydra propio de Blender.

*Referencias:* [43], [47] del §10; especificación OpenFX en openeffects.org; Cryptomatte en github.com/Psyop/Cryptomatte; código: `integrations/ofx/src/AtheneaOfx.cpp`, `integrations/ofx/src/RenderServer.cpp`, `shaders/athenea/technique/crypto_ids.slang`, `modules/usd/src/Export.cpp` (`writeParticleFieldStage`), `integrations/blender/athenea_hydra/__init__.py`; `docs/decisions.md`, secciones «The OpenFX plugin: athenea's gaussians in Nuke (18 Oct)», «Layers and the sky (19 Oct)» y «A pixel says which prims it saw: Cryptomatte».
:::

::: {.nota}
**Formato web: PCA por clusters, cuantización vectorial, registro de 40 bytes, WebGPU.** El *PCA por clusters* (CPCA, Sloan et al. 2003, [15]) agrupa los vectores de transfer con k-means y, dentro de cada cluster $c$, aproxima cada vector como la media más una combinación de $M$ direcciones principales: $\mathbf{t} \approx \bar{\mathbf{t}}_c + \sum_{j=1}^{M} q_j\,\mathbf{d}_{c,j}$. Con 112 valores por Gaussian y $M$ pequeño (ocho pesos por mitad en el formato compacto), la memoria por Gaussian cae un orden de magnitud y la reconstrucción es un producto matriz-vector en la GPU. La *cuantización vectorial* (VQ) sustituye cada máscara de 256 bits por el índice de la entrada más cercana de un *codebook* (aprendido con k-means bajo distancia de Hamming). El registro compacto de 40 bytes agrupa posición cuantizada en la caja de su chunk (3 × u16), albedo, opacidad, escala logarítmica, cuaternión empaquetado, normal octaédrica, curvatura, índice de material, cluster, pesos CPCA e índice de sombra, en planos de bytes (SoA) para comprimir mejor. *WebGPU* ([46]) es la API gráfica moderna de la web (sucesora de WebGL), con compute shaders, que permite ejecutar el mismo tipo de kernels en el navegador de un iPhone.

*Referencias:* [11], [15], [23], [46] del §10; Gersho y Gray, *Vector Quantization and Signal Compression* (1992); el player sparkwebGPU (demos públicas en https://sparkwebgpu.lucab.co.uk) lee el registro compacto, reconstruye el transfer CPCA en la GPU y decodifica el contenedor `.athc` v3; código en athenea: `shaders/athenea/lod/transfer_codec.slang`, `shaders/athenea/lod/shadow_vq.slang`, `modules/lod/src/Athc.cpp`.
:::

::: {.nota}
**Palabras clave.** Resumen del vocabulario que el artículo da por conocido y que estas notas definen en su primera aparición: *relight baking* (precalcular el transporte para reiluminar), *precomputed radiance transfer* (el transporte en SH, [14]), *spherical harmonics* (la base de funciones sobre la esfera), *Hydra* (la arquitectura de render de USD), *cadenas solo en GPU* (pipelines en los que los datos no vuelven a la CPU entre pasos), *composición* (combinar capas de imagen en un compositor como Nuke) y *canales de imagen* (AOVs).
:::
