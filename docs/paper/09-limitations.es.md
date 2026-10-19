<!-- Copyright (c) 2026 jesus luque. -->

# 9. Limitaciones y trabajo futuro

Las limitaciones de abajo se reúnen de las listas de las propias secciones (§2.16, §3.12, §4.12, §5.13, §6.11, §7.13)
y se agrupan por dónde actúan, primero el bake con relight y al final la aplicación a móvil. Cada una se enuncia con la
cifra que la mide, cuando la hay.

::: {.nota}
**Cómo está organizada la lista** — Cada limitación se clasifica por la etapa donde nace: el bake (lo que la nube no puede guardar), la representación (lo que una Gaussian no puede expresar), los assets animados, el motor, el LOD y el player, y por último lo que la evaluación no ha medido. Distinguir *qué limita la representación* de *qué limita la implementación actual* importa al leer cada fila: una limitación de representación (una normal por Gaussian, un campo de rango uno) solo se levanta cambiando lo que se guarda y pagando memoria; una de implementación (la TLAS reconstruida cada frame, la compilación síncrona) se levanta con trabajo de ingeniería sin cambiar el formato.

*Referencias:* §2.16, §3.12, §4.12, §5.13, §6.11, §7.13 del artículo; `docs/decisions.md` (los apartados "Not done" y "Not done, not verified" de cada sección).
:::

## 9.1 Defectos que quedan en el Corvette

Esto es lo que el coche de 1280 px con relight aún muestra frente al path trace.

| defecto | medido | causa | remedio propuesto | § |
|---|---|---|---|---|
| faros oscuros | faros oscuros frente al path trace | el bake del transfer no lleva la emisión de las lámparas | hacer el bake de la emisión de las lámparas y de su luz sobre el coche | 3.12, 4.12 |
| tira cromada del parachoques delantero | un 38 % más clara que el path trace, con el pico más bajo | un tubo de unos pocos píxeles de ancho: cada Gaussian lee su propio espejo con una sola normal | — (abierto) | 3.12, 4.12 |
| reflejo del suelo cercano en la llanta | 0.72 del path trace | — (no analizado más) | — (abierto) | 4.12 |
| reflejos abiertos de la puerta | 1.13 del path trace; el 68 % del exceso donde el espejo da contra el coche | cells de 16 × 16 de unos 11° dejan pasar cielo detrás de un retrovisor o un pilar | cells de 32 × 32 (+96 B por Gaussian) lo reducirían más o menos a la mitad | 4.10.4, 4.12 |
| brillos del capó | a 2.0 px de su posición en el path trace | el giro por píxel extrapola el shape operator del triángulo central | una curvatura ajustada sobre el footprint con muestras del mesh ($p_{90}$ 9.8° frente a 46°) lo reduciría más o menos a la mitad | 4.10.6, 4.12 |
| labio inferior del parachoques | irregular | el footprint de una cell por Gaussian en un borde enrollado | — (abierto) | 3.12 |
| huecos de la grille en los cuts gruesos | 2.24 de las leaves a 640 px, 1.04 a 1280 px | merges de tiras de menos de un píxel; el hueco es una propiedad de la proyección | leaves más finas (hecho a 1280 px); ninguna medida por nodo independiente de la vista lo ve | 6.5, 6.11 |
| molduras finas lejanas | trazos gruesos en todos los cuts | las cotas de error no ven este error | — (abierto) | 6.11 |

::: {.nota}
**Faros oscuros** — El transfer guarda cómo responde cada Gaussian a la luz *entrante* (el cielo, el sol y las luces del stage a través de las open directions); la emisión de una lámpara es un término fuente $L_e$ que el bake no lleva, y su luz sobre el coche (el reflector, la lente, el capó cercano) tampoco. Como las lámparas de un coche están fijas respecto al coche, su contribución es estática en el marco del objeto y podría hacerse bake como un término constante por Gaussian, encendible por grupos de luces (el mecanismo de light groups ya separa la luz de cruce y los pilotos para el light catcher del suelo, §3.9).

**Tira cromada del parachoques** — Un tubo de cromo de unos pocos píxeles de ancho: a lo ancho del tubo la normal gira hasta 180°, y la dirección reflejada gira el doble. Una Gaussian que cubre ese ancho con una sola normal refleja un único punto del entorno en lugar del arco que barre el tubo; el ensanchamiento de Toksvig (§3.7) convierte ese arco en una roughness mayor, que da la media correcta pero aplana el pico (un 38 % más clara y con el pico más bajo). Haría falta una normal que varíe dentro de la Gaussian de forma no lineal (un modelo cilíndrico del tubo) o Gaussians más estrechas que el tubo.

**Reflejos abiertos de la puerta, cells de 16 × 16** — Las open directions son una rejilla octaédrica sobre la esfera: 16 × 16 = 256 cells de unos $4\pi/256\approx0.049$ sr cada una, del orden de 11–13° de lado. Un retrovisor o un pilar más estrecho que una cell deja la cell «abierta» y el reflejo muestra cielo donde el path trace muestra coche (el 68 % del exceso de 1.13). 32 × 32 son 1024 cells de unos 5.6°: 1024 bits en lugar de 256, es decir 32 enteros en lugar de 8, los +96 B por Gaussian de la tabla.

**Brillos del capó** — El giro por píxel extrapola la normal a lo largo de la Gaussian con el shape operator $S$ (la derivada de la normal respecto a la posición sobre la superficie) del triángulo central: $n(x)\approx n_0+S\,x$. Fuera de ese triángulo la curvatura cambia, y el 64 % de los footprints del capó cruzan una arista. Ajustar $S$ por mínimos cuadrados a muestras del mesh sobre todo el footprint baja el error $p_{90}$ de 46° a 9.8° (tabla de la §4.10.6).

**Labio inferior, huecos de la grille, molduras finas** — Un borde enrollado sobre sí mismo tiene dos superficies dentro de una cell. Los huecos de la grille dependen de la proyección: un merge de dos tiras subpíxel llena el hueco entre ellas, y ninguna medida por nodo que no dependa de la vista puede saber cuándo ese hueco es visible. Las molduras finas lejanas tienen el mismo problema: las cotas de error del cut miden cuánto se desplaza la masa, no si una línea fina se ensancha.

*Referencias:* [20], [15], [35] del §10; Cigolle et al., «A Survey of Efficient Representations for Independent Unit Vectors», JCGT 3(2), 2014 (el mapa octaédrico); M. do Carmo, *Differential Geometry of Curves and Surfaces* (1976), para el shape operator. Código: `shaders/athenea/splat/splat_relight.slang`, `shaders/athenea/splat/splat_project.slang`; `docs/decisions.md` (secciones "Step 1: a reflection's occlusion has a direction", "A reflection off a curved strip is the mean over the gaussian (Toksvig)", "A TX transfer's reflection turns across the gaussian (proposals 044, 046, 049)", "A car's lamps switched at run time: light groups (proposals 062, 066)").
:::

## 9.2 La representación y el transfer

- **Campo de rango uno.** El acoplamiento del reflected field al cielo pone en un reflejo el color de un suelo
  iluminado, no su forma, bajo un cielo iluminado desde un lado; con grado 3 resuelve un horizonte a unos veinte grados
  (correlación 0.56 con grado 3, 0.63 con grado 5, 0.85 con un mapa octaédrico de 32 × 32, en simulación) (§4.12).
- **Luces dentro de la escena.** Se extrae un sol por dome. Las lámparas dentro de una escena se sombrean con bits que
  dicen qué direcciones salen de la escena, no cuáles llegan a la lámpara (§4.12).
- **Una normal, un footprint.** Una Gaussian tiene una normal y el footprint de una cell; las tiras finas curvadas y
  los bordes enrollados son donde se nota (§9.1). La correspondencia de coverage supone el solapamiento $2\pi\sigma^2$
  de la lattice de cells, que no tienen los triángulos menores que una cell ni las Gaussians sujetas por bordes
  (§3.12).
- **Vidrio.** Un vidrio fino tintado se convierte como sólido, y una nube conserva un único índice de refracción salvo
  que lleve capas (§3.12). Las Gaussians de lámina fina se suben a un suelo de opacidad de 0.1 por Gaussian, así que
  hacer merge de menos láminas y más grandes hace variar la transmisión de cell a cell; queda pendiente un suelo por
  píxel (§6.11).
- **Tamaño en el device.** La memoria del device limita una pieza a unos 5 M leaves para el merge tree; el
  tope de 4.5 M hizo que las piezas más grandes se convirtieran a una resolución de cámara más gruesa (Bottom a
  3392 px, Part_3 y Front_Bumper_001 a 4480 px; §3.11, §6.9).
- **Instanciado.** Los point instancers anidados solo se expanden en el nivel más interno, y el movimiento creado como
  velocidades se lee en la muestra (§3.12).

::: {.nota}
**Rango uno** — El reflected field guarda, para las direcciones cerradas de cada Gaussian, lo que esas direcciones muestran, *acoplado* al cielo: un valor por Gaussian multiplicado por una función del cielo actual. Como función de dos variables (posición y cielo) es un producto separable $f(x)\,g(\text{cielo})$, es decir de rango uno: cuando el cielo cambia, el reflejo cambia de brillo y de color, pero la *forma* de lo reflejado (dónde está iluminado el suelo) no. Bajo un cielo iluminado desde un lado, el suelo real tiene una zona en luz y otra en sombra; el campo de rango uno pone su color medio.

**Grado de los harmonics** — Los spherical harmonics hasta grado $\ell$ forman una base limitada en banda: la menor estructura angular que representan mide del orden de $\pi/\ell$, y un escalón (un horizonte) reconstruido con ellos se ensancha y oscila (fenómeno de Gibbs). Las correlaciones simuladas (0.56 con grado 3, 0.63 con grado 5, 0.85 con un mapa octaédrico de 32 × 32) muestran que subir el grado rinde poco y que un mapa explícito rinde mucho, a un coste de memoria mucho mayor. La salida intermedia sería un rango mayor: varias funciones base compartidas (como CPCA [15]) en lugar de una.

**Luces dentro de la escena** — Las open directions dicen qué direcciones *salen de la escena*, no si una dirección llega a una lámpara situada dentro de ella. Para un sol o una lámpara lejana es lo mismo; para una lámpara rodeada de objetos, un objeto más allá de la lámpara en la misma dirección la «sombrea» indebidamente. Además, de cada cúpula se extrae un solo sol (los texels más brillantes forman un disco que se ilumina como luz analítica). Lo levantaría guardar profundidad por cell (a qué distancia se cierra la dirección) o hacer bake de la visibilidad de cada lámpara fija.

**Una normal, un footprint, y la coverage** — La coverage de una lattice de Gaussians: una Gaussian no normalizada $g(r)=e^{-r^2/2\sigma^2}$ integra $2\pi\sigma^2$ en el plano, y en una lattice de paso 1 cell con $\sigma=1$ cell la suma en cada punto es aproximadamente $2\pi$. Con opacidad $a$ por Gaussian, la transmitancia es $\prod_i(1-a\,g_i)\approx e^{-a\sum_i g_i}=e^{-2\pi a}$, y de ahí sale la opacidad que reproduce la coverage del material. Un triángulo menor que una cell, o una Gaussian sujeta en un borde, no tiene vecinas que completen esa suma, y su coverage queda corta.

**Vidrio** — Un vidrio fino tintado convertido como sólido refracta como una lente gruesa; una nube con un índice de refracción único no puede mezclar materiales de índice distinto salvo por capas. El suelo de opacidad de 0.1 por Gaussian de lámina fina se suma de forma no lineal: dos láminas de transmisión $0.9$ dan $0.81$, una lámina fusionada que se sube a 0.1 da $0.9$, y la transmisión varía de cell a cell según cuántas láminas haya. Un suelo aplicado por píxel, tras componer, la fijaría.

**Tamaño en el device e instanciado** — El merge tree de una pieza se construye entero en la GPU: con ~5 M leaves el bosque de trabajo no cabe, y las piezas mayores se convirtieron a una resolución menor. Construir por partes y unir los árboles levantaría el tope. En el instanciado, solo el point instancer más interno se expande, y el movimiento autorizado como velocidades se lee en el instante de la muestra en lugar de extrapolar $p+v\,\Delta t$.

*Referencias:* [14], [15], [16], [17], [43] del §10; Ramamoorthi y Hanrahan [17] sobre la banda limitada de la iluminación en SH; código: `shaders/athenea/technique/env_sun.slang`, `shaders/athenea/usd/transfer_field_fill.slang`, `shaders/athenea/usd/mesh2splat_sheet.slang`; `docs/decisions.md` (secciones "Step 2: what the closed directions show", "Step 3: degree 3, and a maximum level to measure against", "Step 4: lights that are not the sky", "The sun comes out of the sky and is lit as a light", "A thin sheet of glass is drawn as a layer", "A windscreen modelled as one surface is a sheet (`--thin-glass`)", "A converted surface covers what its opacity says").
:::

## 9.3 Assets animados

- Cuatro influencias por Gaussian truncan las puntas de las alas del gorrión (el 27 % de los vértices del ala llevan
  más); la proporción de Gaussians truncadas no se ha medido (§5.3, §5.13).
- No hay skinning por cuaterniones duales para las nubes, ni retargeting, ni mezcla de clips; de varios meshes con
  skinning con distintas transformaciones de prim solo se conserva la del primero (§5.13).
- La ruta trazada posa una vez por frame, así que no desenfoca una nube con skinning, y las instancias de LoD no llevan
  movimiento (§5.13).
- La visibilidad por parte deja fuera el autosombreado de la propia parte, supone partes rígidas que ocluyen de forma
  independiente, y no se ha hecho su bake por encima de unos 3 M Gaussians por parte; los transfers zonales conservan
  solo la direct half y el autosombreado de la bind pose (§4.12, §5.8, §5.13).
- El error del ajuste zonal en el gorrión, y una medida geométrica de la coverage en una flexión cerrada, no se han
  medido (§5.13).

::: {.nota}
**Cuatro influencias** — En *linear blend skinning* (LBS) cada punto se transforma por $\sum_j w_j M_j$, con $M_j$ la matriz de skinning de la articulación $j$. Una Gaussian guarda las cuatro influencias de mayor peso, renormalizadas; un vértice de pluma del gorrión tiene entre diez y veinticinco, y truncar desplaza la Gaussian respecto al mesh en las puntas. El remedio es guardar más influencias (8 o un número variable con un índice de comienzo), con más memoria por Gaussian.

**Dual quaternions, retargeting, mezcla de clips** — LBS colapsa el volumen en giros grandes (el efecto *candy wrapper*); el *dual quaternion skinning* (Kavan et al., 2008) interpola rotaciones rígidas y lo evita. *Retargeting* es aplicar una animación a otro skeleton; la *mezcla de clips*, interpolar dos animaciones. Ninguna de las tres está; los tres son trabajo de runtime sin cambio de formato.

**Motion blur en la ruta trazada** — El rasterizer convierte el movimiento durante el shutter en un término de covarianza de rango uno (§2.5.3); la ruta trazada posa la nube una vez por frame (refit de la BVH), así que para ella la nube está quieta durante el shutter. Habría que posar en varios instantes o guardar el desplazamiento por Gaussian en la estructura de rayos.

**Visibilidad por partes** — Cada parte rígida lleva un campo de visibilidad que mide cómo ocluye al resto; la oclusión de varias partes se compone como producto, $V\approx\prod_k V_k$, que es exacto solo si las partes ocluyen direcciones disjuntas (si dos partes tapan la misma dirección, el producto la oscurece dos veces). La parte no se sombrea a sí misma con su propio campo, y los transfers zonales conservan solo la direct half y el autosombreado de la bind pose.

*Referencias:* [16], [41], [42], [43] del §10 (UsdSkel); L. Kavan, S. Collins, J. Žára, C. O'Sullivan, «Geometric Skinning with Approximate Dual Quaternion Blending», ACM TOG 27(4), 2008. Código: `modules/geom/src/Skinner.cpp`, `modules/usd/include/athenea/usd/StageRenderer.h` (`holdBindPose`, que documenta las cuatro influencias), `modules/technique/src/SplatVisibility.cpp`, `shaders/athenea/usd/transfer_zonal_io.slang`; `docs/decisions.md` (secciones "A gaussian carries the joints its triangle carries", "A cloud shadows itself by part: baked once, read every frame, no ray", "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame", "A posed cloud refits its ray tracing structure").
:::

## 9.4 El motor

- Los splats todavía no están dentro de los rayos secundarios del path tracer: con meshes, los splats se componen por
  profundidad sobre las superficies del path trace y proyectan sombras sobre ellas, pero un rayo de bounce no los ve
  (§2.16).
- El blur de movimiento y de lente del rasterizer son Gaussians ajustadas por momentos, hasta un 38 % demasiado
  picudas para estelas largas (§2.16).
- Las Gaussians de unas centésimas de píxel desaparecen del raster; el LOD es el remedio (§2.16).
- La visibilidad de meshes por raster paga unos 8 ms de tiempo de host por cada 1800 draws; la compilación de
  materiales es síncrona; la TLAS se reconstruye cada frame; la distorsión de lente más allá de $k_1, k_2$ no se aplica
  (§2.16).
- La ruta trazada lee el transfer de una nube girada frente a un cielo que se queda en el sistema del mundo (§4.12).
- **Hosts.** En Nuke un nodo OpenFX da una capa por nodo, porque Nuke solo acepta de él sus propios planos; el manifest
  del Cryptomatte se queda en el plugin, porque OpenFX no lleva metadatos, así que un matte se elige por id; la cámara
  son knobs enlazados y no la cámara del propio host; las Gaussians no llevan normales ni vectores de movimiento como
  capas. El add-on de Blender está construido sobre un motor anterior y solo funciona en macOS (§2.14).

::: {.nota}
**Splats y rayos secundarios** — El path tracer de meshes y el ray tracer de Gaussians son estructuras distintas: una BVH de triángulos y otra de partículas. Con meshes en la escena, el motor compone los splats por profundidad sobre la imagen del path trace y los usa para sombras, pero un rayo de bounce que sale de un mesh solo consulta la BVH de triángulos. Integrarlos exige que el integrador, en cada vértice, consulte también las partículas y acumule su transmitancia a lo largo del segmento, como hace 3DGRT [7] para escenas solo de partículas.

*Referencias:* [7] del §10; código: `modules/technique/src/PathTracer.cpp`, `modules/render/src/GaussianRayTracer.cpp`; `docs/decisions.md` (sección "Splats shadow meshes in the path tracer").
:::

::: {.nota}
**Por qué un 38 %** — Una Gaussian de desviación $\sigma$ arrastrada a velocidad constante una longitud $L$ es su convolución con una caja de anchura $L$. El rasterizer la sustituye por una Gaussian con la misma varianza, $\sigma^2+L^2/12$ (la varianza de una caja de anchura $L$ es $L^2/12$). Para estelas largas ($L\gg\sigma$), el perfil real es una meseta de altura $1/L$ y la Gaussian igualada tiene pico $1/\sqrt{2\pi L^2/12}$. El cociente es
$$\frac{1/\sqrt{2\pi L^2/12}}{1/L}=\sqrt{\frac{12}{2\pi}}=\sqrt{\frac{6}{\pi}}\approx1.382 ,$$
un 38 % más picuda: la cifra de la limitación sale exactamente de igualar momentos. Lo mismo vale para el bokeh (un disco). El remedio es dibujar varias Gaussians a lo largo de la estela (más coste) o un kernel de meseta analítico.

**Gaussians de centésimas de píxel** — Una Gaussian proyectada de radio mucho menor que un píxel queda por debajo del umbral de opacidad o de tamaño del rasterizer y no contribuye, aunque la suma de muchas de ellas debería cubrir el píxel. Mip-Splatting [4] lo trata con un filtro 2D de tamaño mínimo; aquí el remedio es el LOD, que sustituye miles de Gaussians diminutas por un merge del tamaño del píxel.

**Coste de host, compilación, TLAS y lente** — La visibilidad raster emite un draw por mesh y paga unos 4.4 µs de CPU por draw; el remedio estándar es el dibujo indirecto con la lista de draws generada en la GPU. Compilar un material en el hilo de render bloquea el frame en que aparece. Reconstruir la TLAS cada frame cuesta proporcional al número de instancias; un *refit* cuando solo cambian transformaciones es más barato. La distorsión de lente es el modelo radial $p'=p\,(1+k_1r^2+k_2r^4)$; el modelo de Brown–Conrady añade $k_3$ y términos tangenciales $p_1,p_2$.

**Nube girada en la ruta trazada** — El rasterizer gira la consulta del cielo al marco del objeto; la ruta trazada lee el transfer en el marco del objeto contra un cielo fijo en el mundo, así que un objeto girado ve el cielo girado al revés. Basta con rotar la dirección de consulta (o los coeficientes SH, con las matrices de rotación de Wigner) por la inversa de la transformación del objeto.

**Hosts** — OpenFX es una API de efectos de imagen: un nodo devuelve las imágenes que el host acepta, sin metadatos arbitrarios, así que el manifest de Cryptomatte (la tabla de hash a nombre) no viaja y Nuke solo toma las capas propias del nodo. La cámara del host no se expone a un efecto, y por eso se enlaza por knobs.

*Referencias:* [4], [47] del §10; D. C. Brown, «Decentering Distortion of Lenses», Photogrammetric Engineering 32(3), 1966; J. Ivanic, K. Ruedenberg, «Rotation Matrices for Real Spherical Harmonics», J. Phys. Chem. 100(15), 1996; especificación de OpenFX (openeffects.org). Código: `modules/render/include/athenea/render/Camera.h` (`distortionK1`, `distortionK2`), `modules/world/src/RayTracingScene.cpp`, `shaders/athenea/splat/sky_turn.slang`, `integrations/ofx`; `docs/decisions.md` (secciones "A splat under a shutter: one rank-one term, not five passes", "Depth of field in the rasteriser: the other convolution", "The camera's lens", "The OpenFX plugin: athenea's gaussians in Nuke (18 Oct)", "Layers and the sky (19 Oct)", "A pixel says which prims it saw: Cryptomatte").
:::

## 9.5 LOD, streaming y el player

- **Ponderación de la oclusión.** El codebook compartido se reparte por entradas usadas, no por la frecuencia con que
  una vista descarta un nodo; ponderar por una órbita de vistas en el bake debería llevar el cull por encima del 84 %
  del ideal (§6.11).
- **Sin portar.** La regla de caras traseras de closed shells y el cut por error viven en una rama de desarrollo; el
  `CutSelector` del motor principal no ha cambiado (§6.11).
- **Residencia.** Las pages tienen unas 4× el cut porque los nodos necesarios y los descartados por la vista comparten
  cada page; el player separa la traversal de la carga útil en lugar de encoger las pages (§6.8, §6.11).
- **Opacidad de los merges en el player.** El player dibuja la masa de cada merge, mientras que el rasterizer de athenea
  conserva la coverage por mínimos cuadrados; el ajuste de coverage es la mejor opción salvo en la grille (§6.4).
- **El cut en el player.** La traversal compacta no guarda el tamaño de LoD de un merge, así que un cut combinado de
  tamaño y error no tiene con qué protegerse, y el cut por error solo no ganó al cut por tamaño en todas las vistas
  (§7.13).

::: {.nota}
**Ponderar el codebook** — El codebook de oclusión tiene 8192 entradas compartidas; hoy cada entrada vale lo mismo. Si se reparten según cuántas veces una vista real descarta los nodos que la usan (muestreando una órbita de cámaras en el bake), las entradas irían a donde más nodos ahorran. Es un problema de cuantización vectorial ponderada: k-means con pesos de frecuencia [23].

*Referencias:* [23], [35] del §10; código: `shaders/athenea/lod/occlusion_mask.slang`.
:::

::: {.nota}
**Sin portar** — El `CutSelector` del motor principal (`modules/lod/include/athenea/lod/Lod.h`) selecciona el cut con la regla original; el cut por error y la regla de closed shells están en la rama de desarrollo de los merge trees. Es una limitación de integración, no de método.

**Residencia** — Las pages agrupan nodos por banda de error, no por visibilidad, así que una page necesaria trae también nodos que la vista descarta. Encoger las pages mejoraría la proporción a costa de más peticiones; el player separa la traversal (registros pequeños, todos residentes) de la carga útil, de modo que la ineficiencia solo afecta a la parte barata.

**Opacidad de los merges** — Un merge por momentos conserva la masa (la suma de opacidades por área); el ajuste por mínimos cuadrados de §6.4 conserva la coverage, la luz que realmente bloquea el grupo, que es lo que se ve. La masa cuenta dos veces las colas solapadas de las leaves (un exceso del 27 % en la mediana, §6.4), y el ajuste $a^\star=\sum_x G(x)C(x)/\sum_x G(x)^2$ fija la opacidad del merge a lo que sus leaves cubren de verdad. La grille es la excepción medida (41.16 dB con la masa frente a 39.71 con la coverage, de cerca a 393 px): allí los merges vecinos se solapan y se componen como $1-\prod$, de modo que cualquier opacidad de más oscurece todo el parche. El player, un fork de Spark, dibuja la masa ($\min(\text{coverageMass},0.99)$); llevar la coverage al formato web es cambiar un valor por nodo al escribir el tree (`--tree-coverage`).

**El cut en el player** — El registro compacto de traversal (5–11 B) no guarda el tamaño de LoD de un merge, así que el player no puede combinar «error proyectado» con «tamaño en pantalla» para protegerse de merges grandes con error bajo; y el cut por error solo no ganó al cut por tamaño en todas las vistas. Añadir el tamaño cuesta bytes por nodo en el registro de traversal.

*Referencias:* [2], [11], [34], [35] del §10; código: `modules/lod/include/athenea/lod/Lod.h`, `shaders/athenea/lod/lod_cut.slang`, `shaders/athenea/lod/lod_pages.slang`; `docs/decisions.md` (sección "Decimation: the levels of detail, cut once by what a merge loses").
:::

## 9.6 Huecos en la validación

- **Un único asset de producción.** Las cifras a nivel de coche son de un asset, el Corvette, con el gorrión y el peón
  como controles animado y rígido. La generalidad a otros assets no se ha medido.
- **Un único entorno para el coche.** Las comparaciones a nivel de coche frente al path trace son bajo un cielo
  (`autoshop_01_4k`, §1.6). La fidelidad del relight bajo otros cielos y luces locales no se ha medido a la escala del
  coche.
- **El close-up level a 5120 px frente al path trace** está medido sobre el coche entero con una cámara y un cut (0.25 px:
  24.00 dB en el frame, 18.31 dB en el coche, §6.9) y en el logo de cerca en el player (§7.12.3), no a lo largo de una
  órbita cercana.
- **v8.2b en el móvil.** El conjunto de datos final está publicado y medido frente al path trace en el player
  (§7.12.3); su descarga y sus tiempos de frame en el iPhone real no, ni la base de 512 clusters allí (§7.2.2).
- **Los tiempos en device real** existen solo para las rampas de estrés y dos pruebas en el device (§7.13);
  la velocidad del móvil emulado no es la del móvil.

::: {.nota}
**Validez externa** — Las cifras del coche son de un asset y de una cámara. Un método se ajustó mirando este asset: cada paso de la §4 se aceptó porque mejoraba el Corvette frente a su path trace. En términos estadísticos, el Corvette es a la vez el conjunto de ajuste y el de prueba (*in-sample*); el gorrión y el peón son controles, no un conjunto de prueba independiente (*held-out*) de assets de producción. Que los cambios generalicen se sabría convirtiendo assets que no intervinieron en las decisiones, con la misma métrica.
:::

::: {.nota}
**Un solo cielo** — El objetivo del transfer es reiluminar: la nube debe ser correcta bajo cielos y luces que no vio. Medir solo bajo `autoshop_01_4k` comprueba la reproducción del bake bajo su propia iluminación, no la reiluminación. El peón sí se midió bajo dos cielos y con el cielo girado (§4.3, §8.4), y el arnés tiene `--validate-sky` para dibujarlo todo bajo otro cielo (§3.10); falta hacerlo a la escala del coche. Las limitaciones de rango uno y de luces internas (§9.2) son exactamente las que solo aparecen bajo otra luz.

*Referencias:* §3.10 y §4.3 del artículo; `docs/decisions.md` (secciones "Measured: the pawn under two skies (M5 Pro, release)", "Under another sky: what was wrong, and the gate").
:::

::: {.nota}
**Una cámara frente a una órbita** — El tamaño de cell se derivó de `/World/Camera` (`--cell-from-camera`), así que esa vista es la que el bake favorece: medir solo desde ella es medir *in-sample* respecto a la resolución. El close-up level existe precisamente para las vistas cercanas, donde una leaf de 1280 px ocupa varios píxeles; una órbita de cámaras cercanas, con su propio path trace por vista, mediría si el nivel cumple ahí. 24.00 dB en el frame y 18.31 en el coche frente a 23.90 / 17.88 del bake de 1280 px dicen que el nivel no empeora la vista de lejos; no dicen cuánto mejora de cerca.

*Referencias:* §3.11 y §6.9 del artículo.
:::

::: {.nota}
**Device real** — Las cifras de tiempo del móvil que se pueden usar son las de la página de estrés y las dos pruebas en el iPhone (§7.12.1); las del iPhone emulado miden la lógica del player con la GPU del escritorio (nota de la §8.1). v8.2b, el conjunto final de 12.77 GB, no se ha descargado ni cronometrado en el teléfono; el tiempo de descarga depende de cuántas pages pida el cut, que solo se conoce midiendo.

*Referencias:* §7.12–§7.13 del artículo.
:::

## 9.7 Trabajo futuro

En el orden que sugieren las medidas:

1. Hacer el bake de la emisión de las lámparas en el transfer, para que los faros y su luz sobre el coche coincidan con
   el path trace.
2. Ajustar la curvatura sobre el footprint de cada Gaussian con muestras del mesh, para reducir a la mitad el
   desplazamiento de los brillos del capó y estabilizar el giro por píxel.
3. Pasar a cells de open directions de 32 × 32 donde un coat nítido refleja el propio coche (la clase abierta de
   la puerta).
4. Tratar los tubos de cromo finos y los labios enrollados (la tira y el labio inferior del parachoques), el reflejo
   del suelo cercano en la llanta y las molduras finas lejanas.
5. Medir el coche con relight frente al path trace bajo entornos variados y luces locales, y el close-up level a
   5120 px a lo largo de una órbita cercana del coche entero.
6. Para los assets animados: más de cuatro influencias, oclusión entre partes en poses distintas de la bind pose, y
   bakes de visibilidad por encima de 3 M Gaussians por parte.
7. Ponderar el codebook de oclusión por una órbita de vistas, y portar el cut por error y la regla de closed shells
   al `CutSelector` del motor principal.
8. En la aplicación a móvil, medir v8.2b en el iPhone real, de cerca y de lejos, con la base de 512 clusters.
9. Dibujar los canales de §4.11 para muchos assets, cielos y cámaras, alineados con los frames con relight, y aprender
   de ellos a predecir los canales del transporte o la luz con relight a partir de entradas que no los llevan -- un
   adaptador aprendido sobre un modelo de imagen que maneje la luz de una captura o de un render como lo haría el
   bake.

::: {.nota}
**El orden** — La lista sigue el tamaño del defecto medido y lo directo del remedio. Los puntos 1–4 atacan los defectos de la §9.1 con remedios ya cuantificados (la curvatura por mínimos cuadrados reduce el error $p_{90}$ de la normal de 46° a 9.8°; las cells de 32 × 32 reducirían el exceso de la clase abierta a la mitad con +96 B por Gaussian). El punto 5 cierra los huecos de validación de la §9.6. Los puntos 6–8 son de alcance (animación, LOD, móvil).

**El punto 9** — Propone usar el bake como generador de datos: dibujar los canales de parámetros de la §4.11 (normales, roughness, open directions, transfer) alineados píxel a píxel con frames con relight, para muchos assets, cielos y cámaras, y entrenar con ellos un adaptador sobre un modelo de imagen preentrenado, de forma que el modelo prediga el transporte de luz a partir de una imagen que no lo lleva (una captura o un render sin bake). Es la misma idea que los adaptadores de control sobre modelos de difusión (ControlNet, Zhang, Rao y Agrawala, 2023) y que los trabajos recientes de reiluminación con modelos de difusión (por ejemplo DiLightNet, Zeng et al., 2024, o Neural Gaffer, Jin et al., 2024), con la diferencia de que aquí los pares de entrenamiento serían físicamente exactos por construcción.

*Referencias:* §4.11 del artículo; L. Zhang, A. Rao, M. Agrawala, «Adding Conditional Control to Text-to-Image Diffusion Models», ICCV 2023; C. Zeng et al., «DiLightNet: Fine-grained Lighting Control for Diffusion-based Image Generation», SIGGRAPH 2024; H. Jin et al., «Neural Gaffer: Relighting Any Object via Diffusion», NeurIPS 2024. Código: la herramienta `render` del servidor MCP (`modules/mcp`); `docs/decisions.md` (sección "A transfer the stage turns, and the gaussians' parameters drawn (19 Oct)").
:::
