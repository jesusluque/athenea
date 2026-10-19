<!-- Copyright (c) 2026 jesus luque. -->

# 6. LOD: merge trees, cuts por error, pages y oclusión

Una nube obtenida por bake desde un mesh a 1280 px contiene 8.7 millones de Gaussians para un solo coche. El close-up
level, con bake a 5120 px, contiene 74.5 millones. Las vistas van desde el asset entero a lo lejos hasta un emblema de
cerca, y una nube de ese tamaño no puede dibujarse ni mantenerse residente entera: lo que se dibuja y lo que está
residente debe seguir a la vista, bajo un budget de nodos y uno de memoria, con los datos cargados por streaming
en unidades. Esta sección describe primero el LOD original del motor (§6.1), hecho para escenas capturadas, y explica
por qué no escala a esas nubes. Después trata la estructura que lo sustituye: merge trees por pares por pieza (§6.2),
un único cut por un error de pantalla acotado (§6.3), la opacidad que recibe un merge (§6.4) y una métrica de spill que
se evaluó y no se adoptó (§6.5), dos culls dependientes de la vista (§6.6, §6.7), pages ordenadas por error (§6.8), el
close-up level a 5120 px (§6.9) y las medidas (§6.10). Todo kernel nombrado aquí es un kernel de cómputo en Slang. La
CPU solo lee ficheros y lleva la contabilidad (§2).

::: {.nota}
**Level of detail (LOD)** — Familia de técnicas que sustituye una representación por otra más barata cuando la diferencia no es perceptible desde la vista actual. En mallas, la tradición va de los niveles discretos (varios meshes por objeto, elegidos por distancia) a las representaciones continuas o jerárquicas (*progressive meshes* de Hoppe, 1996; simplificación por *quadric error metrics* de Garland y Heckbert, 1997; el libro de referencia es Luebke et al., *Level of Detail for 3D Graphics*, 2002). En Gaussian splatting el problema es el mismo, pero la primitiva es una Gaussian 3D con opacidad: sustituir $k$ Gaussians por una sola obliga a decidir qué Gaussian «representa» al grupo (aquí, por *moment matching*, §6.1–6.2) y cuándo es aceptable dibujarla (aquí, por un error proyectado en píxeles, §6.3).

**Por qué hace falta aquí** — 8.7 M de Gaussians por coche a 1280 px y 74.5 M a 5120 px superan dos límites distintos: el de dibujo (el rasterizer ordena y compone cada Gaussian en cada tile que toca, con un coste aproximadamente lineal en el número de Gaussians visibles) y el de memoria (cada Gaussian con transfer TX ocupa cientos de bytes; los 148.9 M de nodos del close-up level ocupan 87.4 GB en disco, §6.9). Por eso el texto separa dos *budgets*: uno de **nodos** dibujados por frame (coste de render) y otro de **memoria** residente (lo que cabe en el device o en un teléfono). Lo que está residente se carga por **streaming**, es decir, por unidades (chunks en §6.1, pages en §6.8) que se traen bajo demanda según la vista.

**Cut** — Término central de la sección: en un árbol de niveles, un *cut* es un conjunto de nodos tal que cada leaf tiene exactamente un antepasado (o ella misma) en el conjunto. Dibujar un cut equivale a dibujar toda la nube a una resolución variable por zonas. Las secciones siguientes construyen el árbol (§6.2), eligen el cut (§6.3), y deciden qué partes del árbol hay que tener cargadas para poder elegirlo (§6.8).

**Alternativas en la literatura** — *Hierarchical 3D Gaussians* (H3DGS, Kerbl et al. 2024, [2]) construye una jerarquía sobre una escena capturada y elige un cut por granularidad proyectada; *Octree-GS* ([8]) organiza anclas en niveles de octree elegidos por distancia; LODGE (Kulhánek et al., 2025) combina niveles por distancia y partición en chunks para escenas grandes en hardware modesto; Nanite (Karis et al. 2021, [35]) usa un DAG de clusters de triángulos con error monótono. El diseño de esta sección toma de [2] la representación por momentos, de NanoGS ([9]) el merge por pares y de Nanite la selección paralela por error monótono y la paginación por error.

**Regla de la CPU** — «La CPU solo lee ficheros y lleva la contabilidad» es la regla del motor (`CLAUDE.md` del repositorio: *No CPU arithmetic on data*): ordenaciones, merges, culling y comparación de imágenes son kernels de cómputo; el host solo lee contadores e histogramas.

*Referencias:* [2], [8], [9], [35] del §10; Hoppe, H. (1996) *Progressive Meshes*, SIGGRAPH; Garland, M., Heckbert, P. (1997) *Surface Simplification Using Quadric Error Metrics*, SIGGRAPH; código: `modules/lod/src/Lod.cpp`, `shaders/athenea/lod/`; `docs/decisions.md` (secciones «Levels of detail», «Decimation: the levels of detail, cut once by what a merge loses», «Simplify: fewer gaussians by pairs, and surfels»).
:::

Notación. Un nodo $n$ es una leaf (una Gaussian del bake) o un merge. $\mu_n$ es su centro y $\Sigma_n$ su
covarianza, con escalas $s_n = (s_x, s_y, s_z)$ y opacidad $o_n$. $\mathrm{par}(n)$ es su padre. $\tau$ es el umbral
en píxeles, $f$ es la focal en píxeles a distancia unidad (`pixelsPerUnit`) y $e$ es el ojo.

::: {.nota}
**Gaussian 3D** — Cada splat es una densidad $G(x)=\exp\!\big(-\tfrac12 (x-\mu)^\top\Sigma^{-1}(x-\mu)\big)$ multiplicada por una opacidad $o\in[0,1]$. La covarianza se factoriza como $\Sigma = R\,\mathrm{diag}(s_x^2,s_y^2,s_z^2)\,R^\top$: las **escalas** $s$ son las desviaciones típicas a lo largo de los tres ejes principales (columnas de $R$, guardada como cuaternión). Una escala exactamente 0 es un **surfel** (disco 2D, [5], [6]).

**Leaf y merge** — Una *leaf* es una Gaussian producida por el bake (§3–4); un *merge* es un nodo interno del árbol que representa a todas las leaves bajo él. $\mathrm{par}(n)$ es su padre en el árbol; una raíz es su propio padre en el fichero (`lodParent`).

**$f$ (`pixelsPerUnit`)** — La focal expresada en píxeles: para una cámara perspectiva con altura de imagen $H$ y campo vertical $\mathrm{fov}_y$, $f = \frac{H/2}{\tan(\mathrm{fov}_y/2)}$. Un objeto de tamaño $L$ a distancia $d$ ocupa aproximadamente $L f/d$ píxeles. Toda la sección convierte magnitudes en metros a píxeles con ese cociente. En el código es `pixelsPerUnit` en `LodParams` (`shaders/athenea/lod/lod_common.slang`) y en `CutParams` (`shaders/athenea/lod/lod_tree_cut.slang`), y multiplica también la escala de la transformación de la instancia.

**$\tau$** — Umbral en píxeles: el máximo error (o tamaño) proyectado que se acepta para dibujar un nodo en lugar de sus hijos. $\tau=0.25$ px significa que el error de usar el merge es una cuarta parte de un píxel de salida.

*Referencias:* [1], [3], [5], [6] del §10; código: `shaders/athenea/lod/lod_common.slang` (`LodParams`), `shaders/athenea/lod/lod_blob.slang` (`Blob`: centro, opacidad, escalas, ejes como filas, color).
:::

## 6.1 El LOD original del motor

**Octree por código de Morton.** `LodBuilder` (`modules/lod`, `shaders/athenea/lod/lod_common.slang`) ordena los
splats por un código de Morton de 30 bits. La cell del octree de un splat en el nivel $r$ son los $3r$ bits altos de
su código, así que las cells de un nivel son tramos de prefijos iguales. Una pasada de fronteras y una suma prefija
las encuentran, sin recorrer ningún tree. Cada cell lleva momentos, siguiendo a Kerbl et al. 2024. Con
$w = o \cdot A_2(s)$, donde $A_2$ es el área de los dos ejes más largos,

$$W=\sum w,\quad S_1=\sum w\,\mu,\quad S_2=\sum w\,(\Sigma+\mu\mu^\top),\quad C=\sum w\,c .$$

::: {.nota}
**Código de Morton (Z-order)** — Se cuantiza cada coordenada del centro, normalizada a la caja envolvente, a 10 bits, y se entrelazan los bits ($x_9y_9z_9x_8y_8z_8\ldots$) en un entero de 30 bits (Morton, 1966). Ordenar por ese código recorre el espacio por una curva de llenado en Z, de modo que puntos próximos en el orden suelen estar próximos en el espacio. La propiedad que se usa aquí es más fuerte: los $3r$ bits altos son exactamente el índice de la cell de un **octree** de nivel $r$ (con $2^r$ cells por eje). Por tanto, tras ordenar, las Gaussians de una misma cell forman un tramo contiguo, y cada nivel del octree es una partición en tramos del mismo array ordenado.

**Pasada de fronteras y suma prefija** — Para el nivel $r$, un kernel marca $b_i = 1$ si el prefijo de $3r$ bits del elemento $i$ difiere del de $i-1$ (`lodBoundaries`); una suma prefija exclusiva de $b$ da a cada elemento el índice de su grupo y el número total de grupos. Es la construcción de octrees de Karras ([32]): todo el árbol sale de un sort y operaciones elementales paralelas, sin punteros ni recorridos.

**Momentos de un grupo** — Con un peso $w$ por Gaussian, se acumulan el momento de orden 0 ($W$, la masa), el de orden 1 ($S_1$, masa por centro) y el de orden 2 ($S_2$). El término $\Sigma+\mu\mu^\top$ aparece porque el segundo momento respecto al origen de una Gaussian es $E[xx^\top]=\Sigma+\mu\mu^\top$. $C$ es el color ponderado (y en el código también los coeficientes SH, las normales y la emisión, `kMomentsHead` en `lod_common.slang`).

**El peso $w = o\cdot A_2(s)$** — Lo que una Gaussian aporta a la imagen es aproximadamente su opacidad por el área que cubre. Para una superficie (discos), el área relevante es la de los dos ejes más largos; usar el volumen ($s_xs_ys_z$) daría peso nulo a un surfel. En el código es `discArea`, $\max(s_xs_y, s_ys_z, s_xs_z)$, el producto de las dos escalas mayores.

**Nota sobre el origen** — La fórmula de $S_2$ escrita aquí es respecto al origen, como en [2]. El código actual toma el segundo momento respecto a la media del propio grupo (§6.2.1 explica el porqué numérico): `lod_common.slang` define $S_2=\sum w\,(\Sigma+(\mu-m)(\mu-m)^\top)$ con $m=S_1/W$.

*Referencias:* [2], [32] del §10; Morton, G. M. (1966) *A computer oriented geodetic data base and a new technique in file sequencing*, IBM; código: `shaders/athenea/lod/lod_morton.slang` (`lodMorton`), `shaders/athenea/common/morton.slang` (`morton30`), `shaders/athenea/lod/lod_boundaries.slang` (`lodBoundaries`), `shaders/athenea/lod/lod_leaf_moments.slang` (`lodLeafMoments`), `modules/gpu/include/athenea/gpu/algo/PrefixSum.h`, `RadixSort.h`; `docs/decisions.md` («Levels of detail» → «Built and cut on the device»).
:::

Los momentos se suman, así que cada nivel más grueso se suma desde sus hijos. Una cell se convierte en una Gaussian
con $\mu=S_1/W$, $\Sigma=S_2/W-\mu\mu^\top$ (barridos de Jacobi en el shader), $c=C/W$ y $o=\min(W/A_2,0.99)$. Los
niveles guardados van desde `coarsestLevel` hasta el nivel más profundo cuyo número de cells es como mucho la mitad
del de splats.

::: {.nota}
**Moment matching** — Dada una mezcla de Gaussians con pesos $w_i$, la única Gaussian que conserva su masa, su media y su covarianza es $\mu=S_1/W$, $\Sigma=S_2/W-\mu\mu^\top$. Es también la Gaussian que minimiza $\mathrm{KL}(p\,\Vert\,q)$ entre la mezcla $p$ y una Gaussian $q$ (la «M-projection» de la teoría de la información; ver Runnalls [34]). Que los momentos se sumen es lo que permite construir los niveles de abajo arriba: los momentos del padre son la suma de los de sus hijos (`lodMergeMoments`), y el resultado no depende de cómo se agrupen las sumas.

**Barridos de Jacobi** — Para pasar de $\Sigma$ a (rotación, escalas) hay que diagonalizar una matriz simétrica $3\times3$. El método de Jacobi aplica rotaciones planas que anulan sucesivamente cada elemento fuera de la diagonal; para $3\times3$ converge en pocos barridos y se implementa sin ramas en un shader (función `jacobi` en `lod_finalize.slang` y en `lod_pairs_final.slang`). Las escalas son las raíces de los autovalores; la rotación, los autovectores.

**Opacidad $o=\min(W/A_2, 0.99)$** — La masa dividida por el área propia del merge da la fracción de esa área que las leaves «cubrían». El tope 0.99 es la convención de 3DGS ([1]), que limita $\alpha$ para que la transmitancia acumulada no llegue nunca a cero. §6.4 muestra que este estimador cuenta dos veces las zonas donde las leaves se solapan.

**Qué niveles se guardan** — `coarsestLevel` (1 por defecto en `LodBuildSettings`, `modules/lod/include/athenea/lod/Lod.h`) es el nivel más grueso. El más fino guardado es el más profundo cuyo número de cells es como mucho la mitad del de splats: por debajo de eso un nivel apenas reduce nada (la mayoría de cells contendría una sola Gaussian), y no compensa la memoria.

*Referencias:* [1], [2], [34] del §10; Golub, G., Van Loan, C. *Matrix Computations*, cap. 8 (método de Jacobi); código: `shaders/athenea/lod/lod_merge_moments.slang` (`lodMergeMoments`), `shaders/athenea/lod/lod_finalize.slang` (`lodFinalize`, `jacobi`), `modules/lod/src/Lod.cpp` (`LodBuilder`).
:::

**Cut.** `CutSelector` dibuja un grupo cuando la arista de su cell, proyectada a la distancia más cercana de la cell,
es como mucho $\tau$ y la proyección de la cell padre supera $\tau$. Una cell hija está dentro de su padre, así que la
prueba es monótona a lo largo de cualquier línea de la raíz a una leaf, y cada lugar se dibuja en exactamente un
nivel. Lo único que se lee de vuelta es un recuento por nivel y una lectura por instancia y frame.

::: {.nota}
**Cut por tamaño en el octree** — Para cada grupo, `projectedEdge` (`shaders/athenea/lod/lod_cut.slang`) calcula la arista de su cell, $\ell_r = \text{extent}/2^r$, por $f$, dividida por la distancia del ojo al punto más cercano de la cell (el ojo se proyecta sobre la caja con un `clamp`). Si el ojo está dentro, la distancia es casi cero y la cell nunca es «pequeña». El grupo se dibuja si es pequeño y su padre no lo es (`lodCutGroups`).

**Monotonía** — Una cell hija está contenida en su padre: su punto más cercano al ojo está a una distancia mayor o igual y su arista es la mitad. Su tamaño proyectado es, por tanto, menor o igual. Recorriendo cualquier camino de la raíz a una leaf, el predicado «pequeño» pasa de falso a verdadero una sola vez. El nodo donde cambia es el único que se dibuja de ese camino: eso es lo que significa «cada lugar se dibuja en exactamente un nivel», y es la misma estructura de demostración que la proposición de §6.3.

**Lecturas de vuelta** — La selección es un kernel por nivel, completamente paralelo; a la CPU solo vuelven contadores (cuántos grupos selecciona cada nivel, para dimensionar el gather) en una única lectura por instancia y frame.

*Referencias:* [2], [32] del §10; código: `shaders/athenea/lod/lod_cut.slang` (`projectedEdge`, `small`, `lodCutGroups`, `lodCutFinest`, `lodCutSplats`), `modules/lod/src/Lod.cpp` (`CutSelector`); `docs/decisions.md` («Levels of detail» → «Built and cut on the device»).
:::

**Chunks, `.athc` y streaming.** Los splats se dividen en chunks de 65 536 splats consecutivos en orden de Morton. Los
niveles fusionados se quedan en el device, y los chunks se cargan por streaming. El formato `.athc` (`.lrtc` en
el tree de lucabRTrender) tiene una cabecera de 104 B, tablas de niveles y chunks, y bloques alineados a 4 KiB para
que un chunk pueda mapearse y traerse por fallo de página por sí solo. `StreamingPool` mapea el fichero. Dos hilos de
carga copian los chunks pedidos, los más pedidos primero. Cada ranura se toma de una ranura libre, del chunk pedido
hace más tiempo o de un chunk pedido menos de la mitad. Un grupo del nivel más fino al que le faltan chunks dibuja su
propia Gaussian fusionada, así que la imagen nunca tiene un agujero. En train_7k (742 k splats, M5 Pro), la
construcción tardó 47 ms y produjo 146 k Gaussians fusionadas. A 8 px el cut dibujó 83 k splats (11 %), con un render
de 3.3 ms frente a 12.6 ms y un error absoluto medio de $3\cdot10^{-3}$. En train_30k, el `.athc` cargado por
streaming en 4 de 16 ranuras escribió EXR idénticos byte a byte y se asentó en 2 cuts.

::: {.nota}
**Chunks** — Tramos fijos de 65 536 ($2^{16}$, `chunkSplats` en `LodBuildSettings`) Gaussians consecutivas en orden de Morton: por la localidad del Z-order, cada chunk es una región compacta del espacio. El tamaño fijo hace que todas las ranuras del almacén del device sean iguales y que no haya fragmentación.

**`.athc`** — Contenedor binario: cabecera de 104 B (`static_assert(sizeof(FileHeader) == 104)` en `modules/lod/src/Athc.cpp`), tablas de niveles y de chunks, y bloques alineados a 4 KiB, el tamaño de página de memoria virtual habitual. La alineación permite mapear el fichero (`mmap`) y que tocar un chunk produzca solo los fallos de página de ese chunk. Los bytes van con el mismo empaquetado que en el device, de modo que cargar es copiar. El escritor escribe `nombre.partial` y renombra, para que un fallo no deje un fichero aparentemente completo.

**`StreamingPool`** — Mapea el fichero; dos hilos de carga (`loaders = 2` en `modules/lod/include/athenea/lod/Athc.h`) copian los chunks pedidos desde el mapeo, de modo que los fallos de página ocurren fuera del hilo del frame. La política de reemplazo, en orden: ranura libre; la del chunk no pedido ahora que se pidió hace más tiempo (LRU); o la de un chunk pedido menos de la mitad que el entrante. Ese factor 1/2 es una **histéresis**: evita que dos chunks de necesidad parecida se expulsen mutuamente en frames alternos.

**Sin agujeros** — Un grupo del nivel fusionado más fino al que le faltan chunks dibuja su propia Gaussian fusionada (el antepasado residente más cercano), así que la imagen degrada en resolución, no en cobertura.

**Las cifras** — *train* es la escena de Tanks and Temples usada en 3DGS ([1]); `_7k` y `_30k` son los checkpoints a 7 000 y 30 000 iteraciones de entrenamiento. 47 ms de construcción para 742 k splats y 146 k Gaussians fusionadas (un 20 % adicional). A 8 px el cut dibuja 83 k (11 %) y el render baja de 12.6 a 3.3 ms con un error absoluto medio de $3\cdot10^{-3}$ (sobre valores de color en $[0,1]$). «Se asentó en 2 cuts»: tras dos frames de cut y carga, ningún chunk más cambia de estado. Los EXR idénticos byte a byte prueban que el streaming no altera el resultado una vez asentado.

*Referencias:* [1], [2] del §10; código: `modules/lod/include/athenea/lod/Athc.h` (`StreamingPool`), `modules/lod/src/Athc.cpp`, `shaders/athenea/lod/lod_cut.slang` (`lodChunkNeeds`, `lodCutFinest`); `docs/decisions.md` («Levels of detail» → «Chunks, `.athc` and streaming», «Measured: streaming (M5 Pro)», «In USD: `AtheneaStreamedAssetAPI`»); `docs/operations.md` §3.3 «A large cloud: levels of detail and streaming».
:::

**Por qué no escala a assets con bake.** Cuatro propiedades de esta estructura no encajan con nubes de decenas de
millones de Gaussians con material bajo un budget compartido.

1. El umbral se fija por instancia y no es un budget. Cada instancia cuesta dos envíos bloqueantes y una lectura
   por frame, y no se comparte ningún budget entre instancias.
2. Las cells del octree ignoran el objeto. Una cell fusiona lo que caiga dentro, incluido un emblema con el panel
   bajo él y el vidrio con la pintura.
3. Un splat por streaming cuesta 132 B, frente a ~55 B del registro compacto (§7.4).
4. Solo los splats van en chunks. Los niveles fusionados, hasta uno con hasta la mitad de cells que splats, están
   siempre residentes, y para millones de leaves ellos solos superan el budget.

::: {.nota}
**1. Umbral por instancia** — En el diseño original cada `LodInstance` lleva su propio umbral y cada instancia hace dos dispatches con espera (contadores y gather) y una lectura. No existe un número global de Gaussians a repartir entre instancias: no se puede decir «dibuja como mucho 250 k en total». El cut por error de §6.3 sí admite un budget global (`--cut-budget`, bisección sobre $\tau$).

**2. Cells que ignoran el objeto** — Una cell del octree es una caja en el espacio: fusiona lo que caiga dentro sin mirar material ni pieza. En una nube de bake con material eso mezcla vidrio con pintura, un emblema con el panel bajo él, o las dos caras de una puerta. El merge tree de §6.2 resuelve esto con guardas por clave de pieza y material.

**3. 132 B por splat** — El formato de streaming guarda el empaquetado del device: posición float4 (16 B), forma (4 uint, 16 B), coeficientes SH en f16 y el índice de grupo, por splat. El registro compacto del player (§7.4) cuantiza y comprime hasta unos 55 B: con decenas de millones de splats, la diferencia es de gigabytes.

**4. Niveles siempre residentes** — Solo los splats van en chunks; los niveles fusionados (hasta $n/2$ cells en el más fino) se suben enteros al abrir el fichero. Para 74.5 M leaves eso serían hasta ~37 M de Gaussians fusionadas siempre en memoria, más que todo el budget de un teléfono. Las pages por error de §6.8 hacen que también los nodos internos se paginen.

*Referencias:* [2] del §10; código: `modules/lod/include/athenea/lod/Lod.h` (`LodInstance`, `CutSelector`), `modules/lod/src/Athc.cpp`; `docs/decisions.md` («Levels of detail» → «Not done yet»).
:::

## 6.2 Merge trees por pares por pieza

`athenea decimate --tree` construye un merge tree binario por merge de pares de vecinos más cercanos mutuos. Es NanoGS
(Xiong et al. 2026) en el device (`lod_pairs.slang`, `lod_pairs_final.slang`, `Lod.cpp`), con guardas adecuadas
a nubes de bake con material (figura 17).

::: {.nota}
**Merge tree binario** — Un árbol donde cada nodo interno es la unión de exactamente dos nodos. Es el resultado de un **clustering jerárquico aglomerativo**: se parte de cada Gaussian como cluster y se unen repetidamente pares de clusters «baratos de unir». El árbol registra el orden de las uniones, y cualquier cut de ese árbol es una simplificación válida de la nube.

**Vecinos más cercanos mutuos** — Cada nodo elige el vecino con menor coste de merge; un par $(a,b)$ se fusiona solo si $a$ elige a $b$ y $b$ elige a $a$. Esta condición permite hacer muchos merges en paralelo sin conflictos (cada nodo está como mucho en un par) y, para costes que cumplen la *reducibilidad* (el coste a un cluster unido no es menor que el mínimo a sus partes), produce el mismo árbol que el aglomerativo voraz secuencial (Walter et al. 2008, *Fast Agglomerative Clustering for Rendering*; el principio viene de la cadena de vecinos más cercanos de Benzécri/Juan, 1982). Con costes que no la cumplen exactamente, como aquí, es una aproximación paralela.

**NanoGS** ([9]) propone este merge por pares mutuos para Gaussians; athenea lo lleva al device (`lod_pairs.slang`: `pairCodes`, `pairBest`, `pairHistogram`, `pairMerge`; `lod_pairs_final.slang`: `pairTreeLayout`, `pairFinalize`) y le añade guardas para nubes con material (§6.2.3). `athenea decimate --tree` registra los merges como árbol en vez de quedarse solo con el resultado al ratio pedido.

**Frente a la alternativa** — El octree de §6.1 agrupa por posición; el merge tree agrupa por coste, que puede incluir color, normal, material y pieza. A cambio, el árbol no tiene niveles regulares: un cut puede contener nodos de 1 leaf junto a nodos de $2^{11}$ (§6.8).

*Referencias:* [9] del §10; Walter, B., Bala, K., Kulkarni, M., Pingali, K. (2008) *Fast Agglomerative Clustering for Rendering*, IEEE Symposium on Interactive Ray Tracing; código: `shaders/athenea/lod/lod_pairs.slang`, `shaders/athenea/lod/lod_pairs_final.slang`, `modules/lod/src/Lod.cpp` (`Decimator`, `PairSettings`); `apps/athenea/src/CmdDecimate.cpp` (`--tree`); `docs/decisions.md` («Simplify: fewer gaussians by pairs, and surfels»).
:::

![Figura 17. Un merge tree y su cut. Los nodos son sumas de momentos; el error ε es monótono hacia arriba y la esfera ρ de cada nodo cubre las de sus hijos. Los nodos se agrupan en pages según el error del padre y llevan un índice a un codebook de oclusión conservador; la traversal de cada frame prueba pages, back faces y oclusión.](figures/fig-lod.es.svg)

::: {.nota}
**Lectura de la figura 17** — Los nodos son sumas de momentos (§6.2.1). El error $\varepsilon$ es «monótono hacia arriba»: nunca decrece de un hijo a su padre (§6.3). La esfera $\rho$ de cada nodo contiene las de sus hijos, de modo que la distancia del ojo a la esfera tampoco crece al subir. Esas dos propiedades garantizan que el cut sea una prueba local por nodo. Las pages (§6.8) agrupan nodos por el error de su padre, y cada nodo lleva un índice a un codebook de máscaras de oclusión (§6.7). La traversal de un frame descarta pages enteras (por error proyectado desde su caja), caras traseras (§6.6) y nodos ocluidos (§6.7).

*Referencias:* [35] del §10 (la misma combinación en Nanite: error monótono, esferas que cubren y streaming por pages); código: `shaders/athenea/lod/lod_tree_cut.slang`, `shaders/athenea/lod/lod_pages.slang`, `shaders/athenea/lod/occlusion_mask.slang`.
:::

### 6.2.1 Los nodos como momentos

Un nodo es un vector de 24 floats: $W$, $W\mu$, $W(\Sigma+(\mu-\bar\mu)(\mu-\bar\mu)^\top)$ (6), $W\cdot\mathrm{sRGB}(c)$, $W n$,
$W\alpha$, $W\cdot$metálico, $W\cdot$transmisión, $W\cdot$especular, la clave, el número de registros, la razón de
grosor ponderada y el índice del nodo en el tree. Aquí $w = o\cdot\max(s_xs_y, s_ys_z, s_xs_z)$ (`discArea`), y
$\alpha$ es la roughness GGX al cuadrado. Un merge suma los dos vectores. Como las sumas son asociativas, los momentos
de un nodo son los de todas sus leaves sea cual sea el orden de los merges. Todos los demás atributos (transfer, bits
de sombra, material, joints) se fusionan una sola vez al final, ponderados por masa sobre el tramo de leaves del nodo
(`lodMergeAttribute`).

::: {.nota}
**El vector de 24 floats** — Es `kNodeFloats` en `lod_pairs.slang`, con la disposición: 0 $W$ | 1–3 $W\mu$ | 4–9 $W\Sigma$ respecto a la media (seis componentes únicas de una matriz simétrica: xx xy xz yy yz zz) | 10–12 $W\cdot$color sRGB | 13–15 $W\cdot$normal | 16 $W\alpha$ | 17 metálico | 18 transmisión | 19 especular | 20 clave (bits) | 21 número de registros | 22 grosor ponderado | 23 índice del nodo en el tree. Todo lo que el coste compara se guarda como suma ponderada por masa, de modo que la media de un nodo es la suma dividida por $W$.

**Color en sRGB** — El color se acumula codificado en sRGB y no en lineal, para que una diferencia oscura sobre claro cuente como se percibe (la codificación sRGB es aproximadamente uniforme perceptualmente; en lineal, los oscuros se comprimen cerca de 0).

**$\alpha$ = roughness$^2$** — La parametrización GGX habitual (Walter et al. [24], Burley [27]): la anchura angular del lóbulo especular crece aproximadamente con $\alpha$.

**Asociatividad** — Como un merge es una suma de vectores, el nodo raíz de un subárbol tiene los mismos momentos sean cuales sean el orden y la forma de los merges dentro de él: el nodo «es» la Gaussian ajustada por momentos de todas sus leaves.

**Atributos fusionados al final** — Transfer TX (decenas de floats por Gaussian), bits de sombra, material, joints de un rig: no se arrastran por cada pasada, sino que se fusionan una vez, ponderados por masa sobre el tramo de leaves del nodo (`lodMergeAttribute` en `lod_attributes.slang`). Como las leaves de un nodo son un tramo contiguo en orden de profundidad (§6.2.4, «Disposición»), el kernel recorre `[lodStart, lodStart + lodSize)`.

*Referencias:* [24], [27] del §10; código: `shaders/athenea/lod/lod_pairs.slang` (`kNodeFloats`, `Node`, `nodeOf`), `shaders/athenea/lod/lod_attributes.slang` (`lodMergeAttribute`, `lodAttributeKey`), `shaders/athenea/lod/lod_pairs_final.slang` (`discArea` vía `lod_blob`); `docs/decisions.md` («Simplify: fewer gaussians by pairs, and surfels»).
:::

El segundo momento se toma respecto al propio centro del grupo $\bar\mu$, no al origen: el de un grupo de leaves a partir
del desplazamiento de cada Gaussian a la media del grupo, el de un grupo más basto a partir de los de sus hijos con el
desplazamiento de la media de cada hijo (el teorema de los ejes paralelos), y el merge de un par igual. Respecto al
origen, $\Sigma = S_2/W - \mu\mu^\top$ es la diferencia de dos números del tamaño de $\mu\mu^\top$, que un float guarda
con unos $5\times10^{-7}$ m² a dos metros del origen -- toda la varianza de una Gaussian de un milímetro de ancho. Las
agujas de los bakes del close-up level sobre triángulos largos y finos (0.93 × 0.043 × 0.004 mm en el paragolpes
delantero, con varianzas transversales de $2\times10^{-9}$ m²) se fusionaban tan finas como las dejaba el redondeo, y el
coste leído de esos mismos momentos las daba casi gratis, así que un cut las elegía y dibujaba sus huecos como grano
brillante. En la esquina delantera izquierda del paragolpes con su bake aislado a 5120 px (301 k Gaussians), el cut a
0.01 px desde una cámara cercana pasó de relMSE $8.8\times10^{-4}$ a $1.3\times10^{-5}$ frente al bake, y el cut a
0.25 px desde `/World/Camera` dibuja 42 719 Gaussians en lugar de 94 351 con algo menos de error ($3.7$ frente a
$3.9\times10^{-4}$).

::: {.nota}
**Cancelación catastrófica** — Un float de 32 bits tiene 24 bits de mantisa: su precisión relativa es $2^{-24}\approx6\times10^{-8}$. A dos metros del origen, $\mu\mu^\top$ vale del orden de $4$ m², y el espaciado entre floats consecutivos cerca de 4 es $2^{2}\cdot2^{-23}\approx4.8\times10^{-7}$ m²: la cifra «$5\times10^{-7}$» del texto. Restar $S_2/W-\mu\mu^\top$ con ambos términos de ese tamaño deja un resultado con error absoluto de ese orden. Una Gaussian de 1 mm de ancho tiene varianza del orden de $(0.5\,\text{mm})^2=2.5\times10^{-7}$ m²: el redondeo se la come entera. Es el ejemplo de manual de Goldberg (1991).

**Teorema de los ejes paralelos (Steiner)** — Si cada hijo $k$ tiene masa $W_k$, media $\mu_k$ y segundo momento centrado $M_k$, el segundo momento del padre respecto a su media $\bar\mu$ es
$$M_p=\sum_k\big(M_k + W_k(\mu_k-\bar\mu)(\mu_k-\bar\mu)^\top\big).$$
Para un par se simplifica a $M_a+M_b+\frac{W_aW_b}{W_a+W_b}\,d\,d^\top$ con $d=\mu_a-\mu_b$ (el término `between` de `pairMerge`). Los desplazamientos son milímetros, y lo que un float pierde sobre ellos también es pequeño en términos relativos. Es la misma idea que el algoritmo de Welford para la varianza en línea.

**Las agujas** — 0.93 × 0.043 × 0.004 mm: Gaussians muy alargadas que el bake a 5120 px deja sobre triángulos largos y finos del paragolpes. La varianza transversal $(0.043\,\text{mm})^2\approx1.8\times10^{-9}$ m² es unas 250 veces menor que el redondeo, así que el merge salía con la anchura que dictaba el redondeo, y el coste de forma (calculado con esos mismos momentos) era casi nulo: el cut las prefería y dibujaba huecos.

**relMSE** — Error cuadrático medio relativo, $\frac1N\sum\frac{(x-y)^2}{y^2+\epsilon}$, la métrica habitual en render Monte Carlo porque no la dominan las zonas brillantes. 8.8e-4 → 1.3e-5 es casi dos órdenes de magnitud. En la vista lejana el árbol corregido dibuja menos de la mitad de Gaussians (42 719 frente a 94 351) con un error algo menor: los costes erróneos mantenían separados merges anchos válidos y dejaban pasar merges degenerados.

*Referencias:* Goldberg, D. (1991) *What Every Computer Scientist Should Know About Floating-Point Arithmetic*, ACM Computing Surveys; Welford, B. P. (1962) *Note on a method for calculating corrected sums of squares and products*, Technometrics; código: `shaders/athenea/lod/lod_pairs.slang` (`pairMerge`), `shaders/athenea/lod/lod_merge_moments.slang`, `shaders/athenea/lod/lod_leaf_moments.slang`, `shaders/athenea/lod/lod_pairs_final.slang` (`pairFinalize`, momentos respecto al primer miembro); `docs/decisions.md` («A merge's second moment about its own centre (13 Oct)»).
:::

La Gaussian fusionada (`pairFinalize`) toma $\mu$ y $\Sigma$ ajustadas por momentos, descompuestas en autovectores
por Jacobi con el eje más delgado en tercer lugar. Fusionar por momentos conserva el segundo momento. Un bloque de
$2\times2$ discos tan anchos como su separación se fusiona por tanto en un disco de 0.58 de esa separación, lo que deja
agujeros visibles de cerca. A lo largo de cada eje ancho $u$, el merge se ensancha así. Sea $v_u$ la varianza
ponderada por área de los centros de los miembros, $c_u$ su anchura media y $d^2 = 12\sqrt{v_uv_v}/(n-1)$ la
separación de una cuadrícula de $n$ miembros sobre esa dispersión. Entonces

$$s_u' = \mathrm{clamp}\!\left(\frac{c_u}{d}\sqrt{12v_u+d^2},\; s_u,\; 2s_u\right).$$

::: {.nota}
**Por qué el moment matching estrecha** — El segundo momento de un parche cubierto uniformemente de lado $L$ es $L^2/12$ por eje, así que la Gaussian ajustada por momentos tiene $\sigma=L/\sqrt{12}\approx0.29L$. Un bloque de $2\times2$ discos separados $D$ cubre un lado $L=2D$; su merge tiene $\sigma\approx2D/\sqrt{12}=D/\sqrt3\approx0.58D$: la cifra del texto. Pero una Gaussian de esa $\sigma$ cae al 60 % de su pico a $1\sigma$ y al 14 % a $2\sigma$: en el borde del bloque (a $D$ del centro, $\approx1.7\sigma$) cubre bastante menos que los cuatro discos originales. Merges vecinos así dejan una rejilla de huecos visibles de cerca (en el suelo de prueba, p99 70 sin ensanchar frente a 1 con el ensanche, `docs/decisions.md`).

**La fórmula del ensanche** — A lo largo de cada eje ancho $u$:
- $v_u$: varianza ponderada por área de los centros de los miembros sobre $u$.
- $d$: separación de una cuadrícula regular de $n$ puntos con esa dispersión. Para una cuadrícula $k\times k$ ($n=k^2$) de paso $d$, la varianza por eje de los centros es $d^2(k^2-1)/12=d^2(n-1)/12$; despejando, $d^2=12v/(n-1)$. Con ejes distintos se usa la media geométrica $\sqrt{v_uv_v}$.
- $\sqrt{12v_u+d^2}$: la longitud real que cubre la fila, $L=kd$, porque $12v+d^2=d^2(k^2-1)+d^2=k^2d^2$.
- $c_u/d$: la razón anchura/separación de los miembros. El merge conserva esa razón respecto al tramo total: $s_u'=(c_u/d)\,L$.

El `clamp` a $[s_u,2s_u]$ impide estrechar por debajo del momento y limita el caso degenerado de miembros casi alineados, en el que $d\to0$ y el ensanche explotaría (el merge se convertiría en un trazo).

**Descomposición** — `pairFinalize` diagonaliza con Jacobi y reordena para que el eje más delgado sea el tercero (`thinAxisThird` en `shaders/athenea/common/packing.slang`); la normal del disco se orienta hacia el lado de las normales de sombreado de los miembros.

*Referencias:* código: `shaders/athenea/lod/lod_pairs_final.slang` (`pairFinalize`, `jacobi`), `shaders/athenea/lod/lod_decimate.slang` (`lodDecimateShape`, la misma regla en la decimación por octree), `shaders/athenea/common/packing.slang` (`thinAxisThird`); `docs/decisions.md` («Decimation: the levels of detail, cut once by what a merge loses» → «Merged wider than the moments»; «Simplify…» → «Coverage»).
:::

La opacidad, con la regla por defecto, es la media de los miembros ponderada por área, limitada a 0.99. La coverage
de masa $W/A(s')$ se guarda como `coverageMass`. Un merge de discos cuyo eje más delgado es menor que 0.12 de su eje
medio se escribe como surfel (escala más delgada exactamente 0).

::: {.nota}
**Opacidad del merge (regla por defecto)** — La media de las opacidades de los miembros ponderada por área ($\sum A_l o_l/\sum A_l$), con tope 0.99. Con el ensanche anterior, el merge cubre aproximadamente lo que cubrían sus miembros, así que heredar su opacidad media es coherente. La alternativa «masa sobre área» se midió peor en la decimación (la pintura del Corvette se emborronaba más allá de sus bordes, porque una suma de discos solapados no es su unión).

**`coverageMass`** — $W/A(s')$: la masa dividida por el área del merge ya ensanchado, sin el tope de 0.99. Se guarda como primvar (`primvars:athenea:splat:coverageMass`) porque el player web la usa como opacidad (§6.4, final).

**Surfel** — Si todos los miembros son discos y el merge sale con el eje más delgado por debajo de 0.12 del medio (`flatTolerance = 0.12` en `PairSettings`), se escribe con esa escala exactamente 0: un disco 2D, como en 2D Gaussian Splatting ([5]) y en los surfels de Pfister et al. ([6]). El rasterizer lo dibuja por intersección rayo-plano en cada píxel; el f16 guarda la escala nula como un logaritmo muy negativo, nunca $-\infty$.

*Referencias:* [5], [6] del §10; código: `shaders/athenea/lod/lod_pairs_final.slang` (`pairFinalize`, `outCoverage`, `surfelCount`), `modules/lod/include/athenea/lod/Lod.h` (`PairSettings::flatTolerance`), `modules/usd/src/Decimate.cpp` (escritura de `coverageMass`); `docs/decisions.md` («Simplify: fewer gaussians by pairs, and surfels» → «A surfel», «Coverage»).
:::

### 6.2.2 El coste

Para los nodos $a, b$ con $\pi = W_a/(W_a+W_b)$ y $\Sigma$ fusionada, el término de forma es la cota de Runnalls de la
divergencia KL por unidad de masa. El área del disco sustituye al volumen, que un surfel no tiene:

$$c_{\text{shape}} = \max\!\Big(0,\ \tfrac12\big[\log I_2(\Sigma) - \pi\log I_2(\Sigma_a) - (1-\pi)\log I_2(\Sigma_b)\big]\Big),$$

donde $I_2$ es el segundo invariante (la suma de los productos de autovalores dos a dos, calculada sin autovalores).
El coste completo es

::: {.nota}
**Divergencia KL y la cota de Runnalls** — Reducir una mezcla de Gaussians fusionando dos componentes $a,b$ en su Gaussian de momentos $m$ pierde información; Runnalls ([34]) da una cota superior de la divergencia de Kullback–Leibler entre la mezcla antes y después:
$$B(a,b)=\tfrac12\big[(w_a+w_b)\log\det\Sigma_m-w_a\log\det\Sigma_a-w_b\log\det\Sigma_b\big].$$
Dividida por la masa total queda la forma del texto, con $\pi=W_a/(W_a+W_b)$. Es siempre $\ge0$ porque $\log\det$ es cóncava sobre matrices definidas positivas y $\Sigma_m\succeq\pi\Sigma_a+(1-\pi)\Sigma_b$.

**Por qué $I_2$ y no $\det$** — $\det\Sigma=\lambda_1\lambda_2\lambda_3$ es el cuadrado del volumen del elipsoide (salvo constantes); un surfel ($\lambda_3=0$) tendría $\log\det=-\infty$. El segundo invariante $I_2=\lambda_1\lambda_2+\lambda_2\lambda_3+\lambda_1\lambda_3$ se reduce a $\lambda_1\lambda_2$ para un disco, es decir, al cuadrado de su área (salvo $\pi^2$). Se calcula sin autovalores como la suma de los menores principales $2\times2$, $I_2=\tfrac12\big[(\mathrm{tr}\,\Sigma)^2-\mathrm{tr}(\Sigma^2)\big]$ (`secondInvariant`). Con $I_2$ la concavidad no está garantizada, de ahí el $\max(0,\cdot)$.

**Escala libre** — El término es un logaritmo de cocientes: no depende del tamaño absoluto. Un coste ponderado por masa fusionaría primero las Gaussians más pequeñas, que son precisamente los detalles; este no.

*Referencias:* [34] del §10; Kullback, S., Leibler, R. (1951) *On Information and Sufficiency*; código: `shaders/athenea/lod/lod_pairs.slang` (`secondInvariant`, `mergeCost`); `docs/decisions.md` («Simplify…» → «The cost»).
:::

$$c = \sigma_{\text{px}}\big(c_{\text{shape}} + w_f\,\phi^2 + w_n(1-\cos\theta_{ab})(1+F_0/\alpha)\big) + w_c\,\lVert\Delta c_{\text{sRGB}}\rVert^2 + w_m\,\Delta_{\text{mat}},$$

donde $\phi$ es el grosor que el merge añade sobre su miembro más grueso. El término $\sigma_{\text{px}} =
\mathrm{smoothstep}(0.5, 2, \text{px})$ anula los términos de forma de los merges que miden menos de medio píxel en la
cámara de la conversión. Valores por defecto: $w_c=20$, $w_n=1$, $w_m=4$, $w_f=4$.

::: {.nota}
**Desglose del coste** (`mergeCost` en `lod_pairs.slang`):
- **$\sigma_{\text{px}}$** — $\mathrm{smoothstep}(0.5,2,\text{px})$, donde px es $\sqrt{\mathrm{tr}\,\Sigma}\,f/d$ del merge visto desde la cámara de la conversión. Si el merge mide menos de medio píxel, su forma no se ve y los términos geométricos no cuentan; a partir de 2 px cuentan enteros. `smoothstep` es la interpolación cúbica $3t^2-2t^3$ entre los dos bordes.
- **$\phi$** — Grosor añadido: el código estima la razón grueso/medio del merge como $\sqrt{\det\Sigma\cdot\mathrm{tr}\,\Sigma/I_2^2}$ (para $\lambda_3\ll\lambda_2$, $\det/I_2\approx\lambda_3$ e $I_2/\mathrm{tr}\approx\lambda_2$) y le resta la del miembro más grueso. Dos discos sobre una superficie curva se fusionan en una lente; este término lo penaliza.
- **Normal** — $(1-\cos\theta_{ab})(1+F_0/\alpha)$: el giro de la normal pesa más cuanto más especular es la superficie, porque un brillo se mueve con la normal en proporción inversa a la anchura del lóbulo. $F_0$ es la reflectancia especular a incidencia normal (canal «especular» del nodo).
- **Color** — $w_c\lVert\Delta c_{\text{sRGB}}\rVert^2$, con la guarda de borde en 0.15 (§6.2.3). Con $w_c=20$, el máximo admisible aporta $20\times0.15^2=0.45$.
- **Material** — $\Delta_{\text{mat}}=\Delta\text{metálico}^2+(\Delta\sqrt\alpha)^2+\Delta\text{transmisión}^2$.

**Valores por defecto** — $w_c=20$, $w_n=1$, $w_m=4$, $w_f=4$: `colourWeight`, `normalWeight`, `materialWeight`, `flatWeight` en `PairSettings` (`modules/lod/include/athenea/lod/Lod.h`). El coste resultante es adimensional; en §6.3 se convierte a metros multiplicándolo por el tamaño del nodo.

*Referencias:* [24], [34] del §10; código: `shaders/athenea/lod/lod_pairs.slang` (`mergeCost`), `modules/lod/include/athenea/lod/Lod.h` (`PairSettings`); `docs/decisions.md` («Simplify…» → «The cost»).
:::

### 6.2.3 Guardas

Un par cuyos miembros difieren en cualquiera de estas formas cuesta $\infty$ (`kNoMerge`):

| guarda | regla | por defecto |
|---|---|---|
| pieza | la clave difiere (prim, joint, lámina, pared fina, metal Schlick) | siempre |
| vidrio | transmisión $>0.5$ en un solo lado | siempre |
| vidrio con vidrio | solo con `--merge-glass` | no |
| espejo | $\alpha<0.01$ en un solo lado | siempre |
| fold | $\cos\theta_{ab} < \cos 60^\circ$ | 60° |
| brillo | $\alpha<0.1$ y $1-\cos\theta_{ab} > \tfrac12\kappa^2\alpha$ | $\kappa=1$ |
| borde de color | $\lVert\Delta c_{\text{sRGB}}\rVert > 0.15$ | 0.15 |
| aguja | aspecto del plano por encima de `maxAspect` y de 1.25× el de los miembros | 1000 (desactivada) |
| trazo | merge más ancho que `maxPixels` en la cámara | no |

::: {.nota}
**Guardas: coste $\infty$** — Una guarda no es un término del coste, sino una prohibición: el par vale `kNoMerge` ($10^{30}$) y nunca se elige. Fila por fila:
- **pieza** — La clave (`lodAttributeKey` en `lod_attributes.slang`) combina el prim de origen (`cryptoObject`), el joint dominante, la marca de lámina (*sheet*, vidrio fino), de pared fina y de metal Schlick. Dos Gaussians con clave distinta no se fusionan nunca, por eso los trees son por pieza (§6.2.5).
- **vidrio** — Transmisión $>0.5$ en un solo lado: vidrio con no vidrio.
- **vidrio con vidrio** — Solo con `--merge-glass`: el parabrisas fusionado 10× daba un moiré irisado, porque cada lámina se dibuja como capa de reflejo (`kSheetAlpha` en `shaders/athenea/splat/splat_project.slang`).
- **espejo** — $\alpha<0.01$ (roughness $<0.1$) en un solo lado.
- **fold** — Normales a más de 60° ($\cos\theta<0.5$, `maxAngle = 60`): un pliegue de la chapa.
- **brillo** — En superficies brillantes ($\alpha<0.1$), si $1-\cos\theta>\tfrac12\kappa^2\alpha$. Como $1-\cos\theta\approx\theta^2/2$, equivale a $\theta>\kappa\sqrt\alpha$: el ángulo entre normales supera la anchura del lóbulo (del orden de $\sqrt\alpha$ rad). Dos normales así ponen el brillo en dos sitios, y su merge lo pondría en uno intermedio (medido: la bola de cromo fusionada 5× brillaba como purpurina).
- **borde de color** — Distancia euclídea sRGB mayor que 0.15 (`edge = 0.15`).
- **aguja** — Aspecto del plano del merge, $\sqrt{\lambda_1/\lambda_2}$ (`aspectOf`, calculado con traza e $I_2$), por encima de `maxAspect` y de 1.25× el de sus miembros. Desactivada por defecto (1000): con 4 el Corvette 10× midió peor.
- **trazo** — Merge más ancho que `maxPixels` px en la cámara: no puede sostener el borde que dibujaban sus miembros.

*Referencias:* código: `shaders/athenea/lod/lod_pairs.slang` (`mergeCost`, `kNoMerge`, `aspectOf`), `shaders/athenea/lod/lod_attributes.slang` (`lodAttributeKey`), `modules/lod/include/athenea/lod/Lod.h` (`PairSettings`: `edge`, `maxAngle`, `highlightCone`, `maxAspect`), `apps/athenea/src/CmdDecimate.cpp` (`--merge-glass`, `--max-angle`, `--highlight-cone`, `--max-aspect`, `--max-pixels`); `docs/decisions.md` («Simplify…», «What no camera sees, and merges no wider than the screen holds (9 Oct)»).
:::

### 6.2.4 Pasadas y la cima relajada

En cada pasada, cada nodo calcula el coste de fusionarse con los 8 vecinos a cada lado a lo largo de un orden de
Morton de los centros. Después repite esto a lo largo de un segundo orden desplazado media cell fina, y se queda con
la pareja más barata. Los pares que se eligieron mutuamente se reparten en un histograma de $\log_{10}$ del coste, y
el host elige el umbral de coste que fusiona tantos pares como permite la pasada, que es como mucho la mitad de los
nodos. El merge se detiene en `treeRatio` (4096×) o cuando ya nada puede fusionarse. Con `relaxTop`, el merge sigue
después con el borde de color (2.0), el ángulo, la guarda de vidrio entre sí, el aspecto y el límite de píxeles
levantados. Las guardas de clave, de vidrio con no vidrio y de espejo se mantienen. Cada uno de esos merges guarda su
coste **negado**, lo que lo marca como merge más allá de las guardas. Esto da a una vista lejana niveles gruesos a los
que cortar: una tapa de faro pasa de 76 486 raíces a 123.

::: {.nota}
**Candidatos por orden de Morton** — Buscar el vecino más barato entre todos es $O(n^2)$. En su lugar, cada pasada calcula el código de Morton de los centros actuales (`pairCodes`), ordena, y cada nodo evalúa los 8 vecinos a cada lado en ese orden (`kWindow = 8`, `pairBest`). El Z-order separa puntos cercanos que caen a ambos lados de una frontera de cell grande; un segundo orden con el origen desplazado media cell fina los vuelve a juntar. En total, hasta 32 candidatos por nodo.

**Umbral por histograma** — Los pares mutuos se reparten en 256 cubetas de $\log_{10}$ del coste (`pairHistogram`). El host lee el histograma (solo contadores) y elige el coste límite que deja fusionar tantos pares como permite la pasada, empezando por los más baratos. Cada par elimina un nodo, así que una pasada reduce como mucho a la mitad.

**`treeRatio` (4096×)** — El merge para cuando quedan $n/4096$ nodos o ninguno más puede fusionarse (todos los pares restantes valen `kNoMerge`).

**La cima relajada (`relaxTop`, activa por defecto)** — Cuando las guardas impiden seguir, el árbol se queda con miles de raíces por pieza, y una vista lejana no tiene niveles gruesos que elegir. Con `relaxTop` el merge continúa con el borde de color en 2.0 (más que la distancia sRGB máxima posible, $\sqrt3\approx1.73$: la guarda queda anulada), sin límite de ángulo (`maxCos = -1`), con vidrio entre sí, sin aspecto y sin límite de píxeles. Se mantienen la clave, vidrio/no vidrio y espejo. Esos merges guardan el coste **negado** (`treeCost[node] = -cost` en `pairMerge`), y el cut los trata aparte (§6.3, factor $g$ y suelo $2r$). Resultado: la tapa del faro pasa de 76 486 raíces a 123.

*Referencias:* [9], [32] del §10; código: `shaders/athenea/lod/lod_pairs.slang` (`pairCodes`, `pairBest`, `pairHistogram`, `pairMerge`), `modules/lod/src/Lod.cpp` (bucle de pasadas, `relaxed`, `kHistogramLo`), `modules/lod/include/athenea/lod/Lod.h` (`relaxTop`, `passes`, `treeRatio`).
:::

**Disposición.** El tree se escribe como un ParticleField. Las leaves van primero, en orden de profundidad, de modo
que cada nodo es un tramo contiguo de leaves. Los merges siguen en el orden en que se hicieron. Cada nodo lleva
`lodParent` (una raíz es su propio padre), `lodStart` (su primera leaf), `lodSize` (su número de leaves) y `lodCost`.

::: {.nota}
**ParticleField** — El esquema de OpenUSD para nubes de partículas Gaussianas ([43]); athenea escribe el árbol como un único ParticleField con primvars adicionales.

**Orden de profundidad** — Las leaves se colocan en el orden de una búsqueda en profundidad del árbol (`pairTreeLayout`: las raíces una tras otra, por suma prefija de sus tamaños; el primer hijo de cada merge en el inicio del padre, el segundo detrás). La consecuencia es que **cada nodo es un tramo contiguo de leaves**, $[\texttt{lodStart},\texttt{lodStart}+\texttt{lodSize})$. Es la representación de intervalos anidados de un árbol (como en un *nested set*), y permite que cualquier kernel recorra las leaves de un nodo sin punteros: la fusión de atributos (§6.2.1), el ajuste de coverage (§6.4) y la clave de pages (§6.8).

**Los cuatro primvars** — `lodParent` (índice del padre en el fichero; una raíz apunta a sí misma), `lodStart`, `lodSize` (1 en una leaf) y `lodCost` (0 en una leaf, negativo en un merge de la cima relajada).

*Referencias:* [43] del §10; código: `shaders/athenea/lod/lod_pairs.slang` (`pairTreeOffsets`, `pairTreeSizes`), `shaders/athenea/lod/lod_pairs_final.slang` (`pairTreeLayout`, `outParent`, `outRun`, `outCost`), `modules/usd/schemas/generatedSchema.usda` (`lodCost`).
:::

### 6.2.5 Muchas piezas: `--passes`, `--merge-glass`, `--object-groups`

En el Corvette entero (5.38 M leaves), el tree puede construirse de una vez, pero escribir sus atributos supera la
memoria del device. `--object-groups K --object-group i` conserva las Gaussians cuyo `cryptoObject` módulo $K$ es
$i$. Los merges nunca cruzan piezas, así que el tree de cada grupo es exactamente el subárbol que el asset entero
tendría para esas piezas. Cada pasada fusiona como mucho la mitad de lo que queda, y un grupo de muchas piezas
pequeñas fusiona pocos pares a la vez. El tope por defecto de 60 pasadas deja un grupo con 116 549 raíces de 323 613
leaves. Con `--passes 1000` llega a 1 298 raíces en 310 pasadas. `--tree-report` muestra dos causas de las raíces
restantes: el vidrio macizo nunca se fusiona (raíces igual a leaves), y los neumáticos fusionan un goteo por pasada
hasta el tope.

::: {.nota}
**`--object-groups K --object-group i`** — Particiona las Gaussians por `cryptoObject mod K` (el identificador de pieza de Cryptomatte, [47]) y construye el árbol de un grupo. Como la guarda de clave impide merges entre piezas, el árbol de una pieza no depende de qué otras piezas haya en el mismo proceso: los $K$ trees juntos son exactamente el árbol del asset entero, partido. Esto permite repartir el trabajo y la memoria.

**Por qué hacen falta tantas pasadas** — El umbral de cada pasada es global: se fusionan primero los pares más baratos de todo el grupo. Una pieza grande y homogénea agota pronto sus pares baratos; muchas piezas pequeñas aportan pocos pares por pasada cada una. Con el tope de 60 pasadas quedan 116 549 raíces de 323 613 leaves (un 36 %); con `--passes 1000`, 1 298 raíces tras 310 pasadas.

**`--tree-report`** — Imprime las raíces restantes por pieza (nombres del manifiesto de cryptomatte). Dos causas: el vidrio macizo, que no se fusiona sin `--merge-glass` (raíces = leaves), y los neumáticos, que fusionan unos pocos pares en cada pasada hasta agotar el tope (el texto constata el goteo sin atribuirlo a una guarda concreta; una geometría muy curvada como la de un neumático, con relieve, es la que más pares lleva cerca de las guardas de fold y de borde).

*Referencias:* [47] del §10; código: `modules/usd/src/Decimate.cpp` (filtro por `cryptoObject`, `--tree-report`), `apps/athenea/src/CmdDecimate.cpp` (`--object-groups`, `--object-group`, `--passes`, `--merge-glass`, `--tree-report`), `modules/lod/include/athenea/lod/Lod.h` (`passes = 60`).
:::

| grupo (coche a 640 px) | raíces antes | con `--merge-glass --passes 4000` |
|---|---|---|
| grupo 2 | 190 490 | 4 237 |
| grupo 5 | 36 508 | 817 |
| coche entero (8 grupos) | — | 17 133 raíces, 5.38 M leaves, 3 582 pages |

::: {.nota}
**Lectura de la tabla** — «Coche a 640 px» es el bake a 640 px de ancho de la cámara de conversión (5.38 M leaves). Con `--merge-glass --passes 4000`, el grupo 2 pasa de 190 490 raíces a 4 237 (45× menos) y el 5 de 36 508 a 817. El coche entero en 8 grupos: 17 133 raíces.

**Comprobación de las cifras** — En un árbol binario cada merge reduce en uno el número de raíces, así que $\text{merges}=\text{leaves}-\text{raíces}$ y $\text{nodos}=2\cdot\text{leaves}-\text{raíces}\approx2\cdot5.38\,\text{M}-17\,133\approx10.74$ M. Con pages de 3000 nodos (§6.8), $10.74\,\text{M}/3000\approx3\,582$: las 3 582 pages de la tabla.

*Referencias:* código: `apps/athenea/src/CmdDecimate.cpp` (`--tree-pages`).
:::

## 6.3 El cut por error, con cotas

**Alternativa considerada: el cut por tamaño.** `--cut-pixels` conserva un nodo cuando su anchura en pantalla
$2\max(s)f/d$ es como mucho $\tau$ y la de ningún antepasado lo es. Los merges más allá de las guardas solo pasan a
$0.25\tau$. Ese cut sigue disponible. El cut por error que sigue, a partir de Nanite, da más PSNR con menos nodos
(tabla al final de esta sección).

::: {.nota}
**Cut por tamaño** — El análogo del cut del octree (§6.1) sobre el merge tree: un nodo se conserva si su anchura en pantalla, $2\max(s)\,f/d$ (diámetro a $1\sigma$ del eje mayor), es como mucho $\tau$ y ningún antepasado cumple lo mismo (`treeCutWidth`, `treeCutKeep`). Los merges de la cima relajada solo pasan a $0.25\tau$ (`relaxedShare = 0.25` en `Decimate.cpp`). Su defecto: mide cuánto ocupa un merge, no cuánto se equivoca. Un merge grande de una pieza homogénea podría dibujarse mucho antes que uno pequeño que mezcla un borde.

**Cut por error, a partir de Nanite** — En Nanite ([35]) cada grupo de clusters guarda el error geométrico de su simplificación y una esfera; un cluster se dibuja si su error proyectado es $\le\tau$ y el de su padre $>\tau$. La misma lógica aparece en el *screen-space error* de los 3D Tiles de OGC/Cesium ($\text{SSE}=\text{geometricError}\cdot H/(2d\tan(\mathrm{fov}/2))$, es decir, $\varepsilon f/d$). El comentario de cabecera de la sección en `lod_tree_cut.slang` cita las dos fuentes.

*Referencias:* [35] del §10; OGC (2019) *3D Tiles Specification 1.0* (`geometricError`); código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCutWidth`, `treeCutKeep`), `modules/usd/src/Decimate.cpp` (`relaxedShare`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-pixels`).
:::

**Error propio.** Sea $r_n=\lVert s_n\rVert$ (la raíz cuadrada de $\mathrm{tr}\,\Sigma_n$) y $\alpha_n =
\max(\text{roughness},0.02)^2$. El error propio de un merge, en metros, es

$$\varepsilon_0(n) = r_n\Big(|\mathrm{cost}_n| + w_c\,s_c(n) + w_n\,\frac{\theta(n)^2}{\alpha_n} + w_s\max(E_n-E_0,0)\Big),\qquad w_c=10,\ w_n=1,$$

y $\varepsilon_0=0$ para una leaf. Los merges más allá de las guardas ($\mathrm{cost}_n<0$) no se miden por su coste,
que no ve el moiré que causan en el vidrio ni el color que arrastran a través de un borde. Por eso se acotan a

$$\varepsilon_0(n) \leftarrow \max\big(g\,\varepsilon_0(n),\ 2r_n\big),\qquad g=4\ (\texttt{--cut-guard}).$$

El término de spill $w_s$ es 0 por defecto (§6.5).

::: {.nota}
**Error propio $\varepsilon_0$, en metros** — El coste del merge (§6.2.2) es adimensional; multiplicarlo por $r_n=\lVert s_n\rVert=\sqrt{\mathrm{tr}\,\Sigma_n}$ (la «talla» del nodo, raíz de la suma de sus varianzas) lo convierte en una longitud que se puede proyectar a píxeles. Los términos:
- $|\mathrm{cost}_n|$: el coste del último merge (en valor absoluto, por los relajados).
- $w_c\,s_c(n)$: la dispersión de color acotada (siguiente párrafo), $w_c=10$.
- $w_n\,\theta(n)^2/\alpha_n$: el cono de normales al cuadrado, relativo a la anchura del lóbulo; $\alpha_n=\max(\text{roughness},0.02)^2$ evita dividir por cero en un espejo perfecto. $w_n=1$.
- $w_s\max(E_n-E_0,0)$: el spill, desactivado ($w_s=0$, §6.5).

**Merges relajados** — Su coste se calculó con las guardas levantadas y no ve lo que rompen (el moiré del vidrio fusionado, el color arrastrado sobre un borde). Por eso su error se multiplica por $g=4$ (`--cut-guard`) y nunca baja de $2r_n$. Con el suelo $2r_n$, $P\ge2r_nf/d$: un merge relajado solo entra en un cut a $\tau$ cuando el nodo mide en pantalla como mucho $\tau/2$ (en términos de $r_n$).

**Implementación** — `treeErrorBounded` en `lod_tree_cut.slang`: `e = size * (|cost| + colourWeight·spread + normalWeight·θ²/α + spillWeight·max(spill − spillFloor, 0))`, y si `cost < 0`, `e = max(guard·e, floorSize·size)`. `Decimate.cpp` fija `colourWeight = 10`, `normalWeight = 1`, `floorSize = 2` y toma `guard` de `--cut-guard`. Una leaf tiene error 0 (`treeErrorOwn`).

*Referencias:* código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorOwn`, `treeErrorBounded`), `modules/usd/src/Decimate.cpp` (parámetros del cut), `apps/athenea/src/CmdDecimate.cpp` (`--cut-error`, `--cut-guard`, `--cut-bounds`, `--cut-spill`, `--cut-spill-floor`).
:::

**Cotas.** El coste de un merge solo cubre el último par que unió. Una cadena de merges, cada uno dentro del borde de
color, puede derivar a través de ese borde. Por eso una dispersión de color $s_c$ y un cono de normales $\theta$ se
suben por el tree con la desigualdad triangular (`treeBoundsUp`), empezando en 0 en las leaves:

$$s_c(p) = \max_{k\in\mathrm{ch}(p)}\big(s_c(k)+\lVert c_k-c_p\rVert\big),\qquad \theta(p) = \max_{k}\big(\theta(k)+\angle(n_k,n_p)\big).$$

Ambas son por tanto cotas sobre cada leaf bajo $p$.

::: {.nota}
**Por qué hacen falta cotas** — El coste de un merge compara solo los dos hijos que une. Una cadena A–B–C–D donde cada paso cambia el color 0.1 respeta el borde de 0.15 en cada merge, pero el nodo final mezcla A y D, que difieren 0.3. Es la deriva de los clusterings aglomerativos con criterio local (el «efecto cadena» del *single linkage*).

**La desigualdad triangular** — Si $s_c(k)$ acota la distancia de color de toda leaf $l$ bajo $k$ al color de $k$, entonces para el padre $p$:
$$\lVert c_l-c_p\rVert\le\lVert c_l-c_k\rVert+\lVert c_k-c_p\rVert\le s_c(k)+\lVert c_k-c_p\rVert,$$
y tomando el máximo sobre los hijos se obtiene $s_c(p)$. Por inducción desde las leaves ($s_c=0$), $s_c(p)$ acota la desviación de cada leaf. Lo mismo vale para las normales con la distancia angular $\angle(n_k,n_p)$, que también es una métrica sobre la esfera: $\theta(p)$ es el semiángulo de un cono alrededor de $n_p$ que contiene la normal de cada leaf. Es la idea de los *normal cones* de la jerarquía de culling de caras traseras (Shirman y Abi-Ezzi, 1993).

**Cálculo en paralelo** — `treeBoundsUp` lanza un hilo por nodo; cada uno eleva el valor de su padre con `InterlockedMax` sobre los bits del float (para floats no negativos, el orden IEEE 754 coincide con el orden de los enteros sin signo). El kernel se repite hasta que el contador `changed` lee 0; el número de rondas es la profundidad del árbol.

*Referencias:* Shirman, L. A., Abi-Ezzi, S. S. (1993) *The Cone of Normals Technique for Fast Processing of Curved Patches*, Computer Graphics Forum (Eurographics); Johannsen, A., Carter, M. B. (1998) *Clustered Backface Culling*, Journal of Graphics Tools; código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeBoundsUp`, buffers `spreadBits`, `coneBits`), `modules/usd/src/Decimate.cpp` (bucle de relanzado).
:::

**Error monótono y esferas que cubren.** El error se hace monótono, y cada nodo recibe una esfera que contiene las
esferas de sus hijos:

$$\varepsilon(n) = \max\big(\varepsilon_0(n),\ \max_{k}\varepsilon(k)\big),\qquad \rho(n)=\max_k\big(\lVert\mu_n-\mu_k\rVert+\rho(k)\big),\quad \rho(\text{leaf})=2\max(s).$$

Cada subida es un kernel lanzado sobre todos los nodos, en el que cada nodo eleva a su padre con `InterlockedMax` sobre
los bits del float. El kernel se relanza hasta que un contador lee cero, lo que da la profundidad del tree en rondas.
El error se proyecta en el punto más cercano de la esfera:

$$P(n) = \frac{\varepsilon(n)\,f}{\max(\lVert\mu_n-e\rVert-\rho(n),\ 10^{-3})}.$$

::: {.nota}
**Error monótono** — $\varepsilon(n)=\max(\varepsilon_0(n),\max_k\varepsilon(k))$: el error de un nodo se eleva al máximo de sus descendientes. Sin esto, un merge barato encima de un merge caro tendría menos error que su hijo, y la selección podría dibujar el padre (bajo $\tau$) a la vez que el hijo (sobre $\tau$ pero con padre «bueno»). Nanite impone la misma condición ([35]).

**Esferas que cubren** — $\rho(\text{leaf})=2\max(s)$ (envolvente a $2\sigma$ del eje mayor) y $\rho(p)=\max_k(\lVert\mu_p-\mu_k\rVert+\rho(k))$: por desigualdad triangular, la bola de centro $\mu_p$ y radio $\rho(p)$ contiene la de cada hijo, y por inducción todas las leaves bajo $p$.

**Error proyectado** — $P(n)=\varepsilon(n)f/\max(d-\rho,10^{-3})$ con $d=\lVert\mu_n-e\rVert$: el error dividido por la distancia al **punto más cercano** de la esfera (la estimación más pesimista para cualquier parte del nodo). Si el ojo está dentro de la esfera, la distancia se fija en 1 mm y $P$ se hace enorme: el nodo se refina.

**Las subidas** — Mismo esquema que las cotas: un kernel por ronda con `InterlockedMax` sobre los bits (`treeErrorUp` sube error y radio; `treeErrorClimb` solo el error, una vez fijas las esferas), relanzado hasta que nada cambia.

*Referencias:* [35] del §10; código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorUp`, `treeErrorClimb`, `projectedError`, buffers `errorBits`, `radiusBits`), `shaders/athenea/lod/lod_pages.slang` (`projectedError`, la misma fórmula).
:::

**Proposición.** Si $P(n)\le\tau<P(\mathrm{par}(n))$ selecciona $n$ (a una raíz le basta la primera desigualdad),
entonces cada leaf se dibuja exactamente una vez. *Demostración.* $\varepsilon$ no decrece al subir. La esfera del
padre contiene la del hijo, así que la distancia a la esfera decrece al subir. Por tanto $P$ no decrece a lo largo de
ningún camino de una leaf a la raíz. Las leaves tienen $P=0\le\tau$, así que el predicado «$P\le\tau$» se cumple en un
prefijo del camino que empieza en la leaf, y exactamente un nodo es el último de ese prefijo. $\square$

::: {.nota}
**Proposición, paso a paso** — Sea el camino $l=n_0,n_1,\ldots,n_m$ (raíz) desde una leaf.
1. *$\varepsilon$ no decrece*: por construcción, $\varepsilon(n_{i+1})\ge\varepsilon(n_i)$.
2. *La distancia a la esfera no crece*: $\lVert\mu_{i+1}-e\rVert\le\lVert\mu_i-e\rVert+\lVert\mu_{i+1}-\mu_i\rVert$ y $\rho_{i+1}\ge\rho_i+\lVert\mu_{i+1}-\mu_i\rVert$; restando, $d_{i+1}-\rho_{i+1}\le d_i-\rho_i$. El $\max(\cdot,10^{-3})$ es monótono, así que conserva la desigualdad.
3. Por 1 y 2, $P(n_{i+1})\ge P(n_i)$.
4. $P(l)=0\le\tau$. El conjunto $\{i: P(n_i)\le\tau\}$ es un prefijo $\{0,\ldots,j\}$ del camino; $n_j$ cumple $P(n_j)\le\tau<P(n_{j+1})$ (o es la raíz). Ningún otro nodo del camino lo cumple: por debajo de $j$ el padre también está bajo $\tau$; por encima, el propio nodo no.

Como el cut es la unión de esos nodos para todas las leaves, cada leaf tiene exactamente un representante: no hay huecos ni dobles dibujos.

**Comparación con §6.1** — Es el mismo argumento que la monotonía del octree, pero el error ya no es la arista de una cell, sino una cantidad del contenido (coste, color, normales) que se hace monótona a propósito.

*Referencias:* [35] del §10; código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorKeep`).
:::

Así que el cut es una sola prueba paralela por nodo (`treeErrorKeep`), sin subir por los antepasados. `--cut-budget
N` biseca $\log\tau$ en $[10^{-4},10^4]$ en 40 pasos para hallar el umbral más pequeño cuyo cut tiene como mucho $N$
nodos.

::: {.nota}
**Una prueba por nodo** — `treeErrorKeep` evalúa en cada hilo `P(i) <= threshold && (root || P(parent) > threshold)`: no hay que subir por los antepasados ni recorrer el árbol, y el coste es $O(\text{nodos})$ por frame con paralelismo total.

**`--cut-budget N` y bisección** — El número de nodos del cut es no creciente en $\tau$ (al subir $\tau$ cada leaf pasa a estar representada por un antepasado igual o más alto). Se puede buscar el menor $\tau$ que cumple el budget por bisección. Se biseca en $\log\tau$ porque el rango abarca 8 órdenes de magnitud ($10^{-4}$ a $10^4$ px); 40 pasos reducen el intervalo logarítmico en un factor $2^{40}\approx10^{12}$, mucho más de lo necesario, pero cada paso es solo un dispatch y la lectura de un contador.

*Referencias:* código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorKeep`), `modules/usd/src/Decimate.cpp` (bisección: `low = log(1e-4)`, `high = log(1e4)`, 40 iteraciones), `apps/athenea/src/CmdDecimate.cpp` (`--cut-budget`).
:::

| configuración | escena | cut por tamaño | cut por error |
|---|---|---|---|
| lejos, 393 px, 0.25 px | seis piezas | 83 k, 34.4 dB | 34 k, 34.9 dB |
| cerca, 426 k nodos | seis piezas | 42.92 / 31.22 dB (frame / grille) | 43.61 / 34.50 dB (prototipo de referencia 43.61 / 34.47) |

::: {.nota}
**Lectura de la tabla** — «Seis piezas» es una escena de prueba con seis piezas del Corvette (incluidas la grille y la tapa del faro). Fila 1: vista lejana renderizada a 393 px de ancho (tamaño de teléfono) con $\tau=0.25$ px: el cut por tamaño necesita 83 k nodos para 34.4 dB y el cut por error 34 k (2.4× menos) para 34.9 dB. Fila 2: vista cercana con ambos cuts ajustados al mismo budget de 426 k nodos; PSNR del frame completo y del recorte de la grille. El cut por error gana 0.7 dB en el frame y 3.3 dB en la grille, la zona difícil. El «prototipo de referencia» es la implementación de investigación previa (43.61/34.47): coincide con el kernel a 0.03 dB.

**PSNR** — $10\log_{10}(\text{MAX}^2/\text{MSE})$; +3 dB equivale a reducir el MSE a la mitad. La PSNR del frame completo incluye el fondo (cielo, suelo) donde no hay error, y diluye las diferencias; la de un recorte (crop) mide la zona que importa. Por eso el texto da ambas.

*Referencias:* código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCutKeep`, `treeErrorKeep`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-pixels`, `--cut-budget`).
:::

## 6.4 La opacidad de un merge: coverage por mínimos cuadrados

La masa sobre la elipse de momentos cuenta dos veces las colas solapadas de las leaves. El exceso mide un 27 % en la
mediana y 2.3× en el p10. En la grille a 393 px, los merges salen más oscuros que sus leaves. `--cut-coverage` (al
hacer el cut) y `--tree-coverage` (al escribir el tree) fijan la opacidad de cada merge a partir de lo que sus leaves
cubren de verdad, así:

1. El plano del nodo lo generan sus dos ejes más anchos $u,v$, con escalas $s_u,s_v$. Una cuadrícula de $8\times8$
   cubre el núcleo $[-1.25,1.25]^2$ en unidades de $(s_u,s_v)$.
2. Cada leaf $l$ bajo el nodo (su tramo, hasta 4096 leaves al escribir el tree y 2048 en un cut) se proyecta sobre
   el plano. Su covarianza en el plano es $(u^\top\Sigma_lu,\ u^\top\Sigma_lv,\ v^\top\Sigma_lv)$ y el desplazamiento
   de su centro es $m_l$. Un tramo más largo conserva su opacidad basada en la masa. Las leaves se componen como lo
   haría el rasterizer: $C(x)=1-\prod_l\big(1-\min(o_l,0.99)\,G_l(x)\big)$.
3. El footprint propio del nodo en la cuadrícula es $G(x)=\exp(-\tfrac12\lVert x\rVert^2)$. La opacidad es el ajuste
   por mínimos cuadrados de $aG$ a $C$:

$$a^\star = \frac{\sum_x G(x)\,C(x)}{\sum_x G(x)^2},\qquad o_n = \mathrm{clamp}(a^\star,0,0.99).$$

::: {.nota}
**Por qué la masa sobreestima** — $W=\sum o_lA_l$ suma las áreas de las leaves como si no se solaparan. Pero las Gaussians de un bake se solapan con sus vecinas (sus colas se extienden más allá del espaciado) y la cobertura real de una pila de capas semitransparentes no es la suma, sino $1-\prod(1-\alpha_l)$. La masa cuenta dos veces lo solapado: un exceso del 27 % en la mediana y de 2.3× en el percentil 10. En la grille a 393 px, el resultado medido es que los merges salen más oscuros que sus leaves: cubren más de lo que cubrían ellas, y tapan más de lo que hay detrás.

**El procedimiento** (`treeCoverage` en `lod_tree_cut.slang`):
1. El plano del nodo lo generan sus dos ejes más anchos; se muestrea con $8\times8$ puntos en $[-1.25,1.25]^2$ en unidades de sus escalas.
2. Cada leaf se proyecta sobre el plano: su covarianza 2D es la restricción $(u^\top\Sigma_lu, u^\top\Sigma_lv, v^\top\Sigma_lv)$ y su centro $m_l$. Se compone como el rasterizer, $C(x)=1-\prod_l(1-\min(o_l,0.99)G_l(x))$: la opacidad acumulada de un *alpha blending* «over» (Porter y Duff, 1984) sin importar el orden.
3. Se ajusta la amplitud $a$ de la Gaussian propia del nodo, $aG(x)$, a $C(x)$ por **mínimos cuadrados**: minimizar $\sum_x(aG-C)^2$, derivar respecto a $a$ e igualar a cero da $a^\star=\sum GC/\sum G^2$. Si $C$ fuera exactamente $kG$, $a^\star=k$: el estimador recupera la amplitud.

Tramos de más de 4096 leaves (al escribir el tree) o 2048 (en un cut) conservan la opacidad por masa: son nodos gruesos de vistas lejanas, donde menos se nota, y el coste del ajuste crece con el tramo.

*Referencias:* Porter, T., Duff, T. (1984) *Compositing Digital Images*, SIGGRAPH; código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCoverage`, `covarianceOf`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-coverage`, `--tree-coverage`).
:::

Se midieron dos alternativas. Normalizar por $\sum G$ en vez de $\sum G^2$ da 0.5 para un merge tan ancho como una
leaf opaca. Un ajuste sobre $\pm2\sigma$ en vez de $\pm1.25\sigma$ da 0.84 a los merges de un panel, porque el borde es
donde los vecinos llevan la superficie, y un guardabarros sale moteado. Las leaves de canto respecto al plano, como
los alambres de una grille vistos de frente, no cubren nada, así que el merge va a 0. Un cargador descarta todo lo que
esté por debajo de 1/255, lo que rompería el tree, así que un merge se escribe como mínimo a

$$o_n \ge 2/255\quad(\texttt{treeCoverageFloor}).$$

::: {.nota}
**Normalizar por $\sum G$** — $\sum GC/\sum G$ es una media de $C$ ponderada por $G$, no una amplitud: incluso si $C$ tiene exactamente la forma de $G$, da un valor menor que su pico (la media de una campana es menor que su máximo). De ahí el 0.5 medido para un merge tan ancho como una leaf opaca.

**Ventana $\pm1.25\sigma$ frente a $\pm2\sigma$** — Fuera del núcleo, lo que cubre la superficie son los vecinos del nodo, no sus propias leaves: un ajuste sobre $\pm2\sigma$ ve las colas medio vacías y baja la opacidad (0.84 en los merges de un panel), y el guardabarros sale moteado porque se ve el fondo a través del centro de cada merge.

**Leaves de canto** — Una leaf cuyo plano contiene la dirección normal del merge (alambres de una grille vistos de frente) se proyecta en una elipse degenerada: no cubre nada en el plano, $C\approx0$ y la opacidad cae a 0. Es correcto para la cobertura en ese plano, pero rompe la carga: el validador del cargador descarta todo registro con opacidad menor que 1/255 (`splat_validate.slang`), y un merge descartado dejaría huérfanos a sus hijos. De ahí el suelo $2/255$ (`kTreeFloor` en `treeCoverageToFile`; `treeCoverageFloor` repone el suelo en merges que el cargador ya había descartado).

*Referencias:* código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCoverage`, `kTreeFloor`, `treeCoverageToFile`, `treeCoverageFloor`), `shaders/athenea/scene/splat_validate.slang`.
:::

El tree se reescribe en su propio orden de fichero, y solo cambian las opacidades de los merges
(`treeCoverageToFile`), porque `lodParent`/`lodStart` son índices del fichero que una reescritura a través del almacén
en orden de Morton revolvería. El ajuste lee los ejes de un Blob como filas. Leídos como columnas dan la rotación
inversa y giran mal cada Gaussian: los merges de un guardabarros salen entonces a 0.59 de su masa donde un ajuste de
referencia da 1.00.

::: {.nota}
**Reescritura en el orden del fichero** — Al cargar, `LodBuilder` reordena el almacén por Morton (`storeOrder` traduce de almacén a fichero). Los índices `lodParent`/`lodStart` apuntan a posiciones del fichero; una reescritura desde el almacén reordenado los invalidaría. Por eso `treeCoverageToFile` escribe un array de opacidades indexado por fichero (NaN en las leaves = no tocar) y solo cambian las opacidades de los merges.

**Filas frente a columnas** — En `Blob` (`lod_blob.slang`) los ejes de la Gaussian son las **filas** de `axes`. Una matriz de rotación es ortogonal: su traspuesta es su inversa. Leer los ejes como columnas aplica la rotación inversa, y cada Gaussian queda girada. Con un error así el ajuste daba 0.59 de la masa en los merges de un guardabarros, donde la referencia daba 1.00. `covarianceOf` lleva el comentario de este fallo.

*Referencias:* código: `shaders/athenea/lod/lod_blob.slang` (`Blob`, `blobOf`), `shaders/athenea/lod/lod_tree_cut.slang` (`covarianceOf`, `treeStoreOf`, `treeCoverageToFile`).
:::

| configuración | qué | coverage | masa |
|---|---|---|---|
| ajuste al hacer el cut (48 k) | grille, 393 px | 39.79 dB (desde 33.87) | — ; leaves 41.59 |
| tree reajustado (49.4 k) | grille, 393 px | 41.28 dB | 39.79 con 48.1 k (`--cut-coverage`) |
| 1 px, cotas, ejes leídos como columnas | seis piezas cerca a 1920 | 40.96 frame / 32.09 grille | tree antiguo 41.11 / 30.09 |
| ejes leídos como filas | seis piezas cerca a 1920 | **43.86** | 42.55 |
| ejes leídos como filas | seis piezas lejos a 393 | 39.95 | 39.66 |
| ejes leídos como filas | grille cerca a 393 | 39.71 | **41.16** |

::: {.nota}
**Lectura de la tabla** — Columnas: PSNR con opacidad por coverage y con opacidad por masa.
- *Ajuste al hacer el cut (48 k)*: en la grille a 393 px la coverage sube de 33.87 a 39.79 dB; dibujar todas las leaves da 41.59 (techo de referencia).
- *Tree reajustado (49.4 k)*: escribir la coverage en el árbol (`--tree-coverage`) da 41.28 dB, frente a 39.79 con el ajuste en el cut (que ajusta solo los nodos del cut).
- *Ejes leídos como columnas*: con el error de §6.4, el nuevo árbol (40.96 frame / 32.09 grille) apenas mejoraba al antiguo (41.11 / 30.09).
- *Ejes leídos como filas*: corregido, la coverage gana en el frame cercano (43.86 frente a 42.55) y en el lejano (39.95 frente a 39.66), pero pierde en la grille cercana (39.71 frente a 41.16). Negrita: la mejor de cada fila.

*Referencias:* código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCoverage`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-coverage`, `--tree-coverage`).
:::

La grille es la excepción. Los merges vecinos se solapan y se componen como $1-\prod$, así que un poco más de
opacidad oscurece todo el parche. El player web (Spark) dibuja la masa de cada nodo
($\min(\text{coverageMass},0.99)$), y el rasterizer del motor conserva la coverage.

::: {.nota}
**La excepción de la grille** — En una superficie continua, los merges vecinos se solapan poco y la cobertura de cada uno es la cobertura de la superficie. En una grille, los merges de tiras paralelas se solapan entre sí, y la composición $1-\prod(1-\alpha)$ es muy sensible a la opacidad cuando hay muchas capas: subir un poco cada $\alpha$ oscurece todo el parche. La opacidad por masa, más baja en ese caso, deja pasar más luz entre tiras.

**Dos lectores, dos opacidades** — El player web (Spark, [11]) dibuja $\min(\text{coverageMass},0.99)$; el rasterizer del motor usa la opacidad ajustada por coverage. El fichero lleva ambas.

*Referencias:* [11] del §10; código: `modules/usd/src/Decimate.cpp` (`coverageMass`).
:::

## 6.5 La métrica de spill, y por qué falló en la grille

En un cut grueso los agujeros de la grille se cierran: en el cut de 0.25 px del coche a 640 px a 1920 px con back-face
culling, la región de suelo cercano lee 2.24× el path trace, frente a 1.18 en las leaves. La causa son merges de dos
leaves de las tiras subpíxel de `Chrome_Part_Grill`, vistas de frente y de 3–6 px de ancho a $\tau=0.25$. Su coste es
~0 porque la coverage de un par, medida en su propio plano, no cambia. Se define un spill sobre la propia cuadrícula
del ajuste de coverage:

$$E_n = \frac{\sum_x \max\big(a G(x) - C(x),\,0\big)}{\sum_x aG(x)}\in[0,1],$$

es decir, la parte de lo que pinta el merge que sus leaves no pintan. Coincide con una implementación de referencia
en NumPy con una correlación de 0.994, y se escribe como `lodSpill`. El cut suma $r\,w_s\max(E-E_0,0)$ al error
propio.

::: {.nota}
**El síntoma** — En un cut grueso los huecos de la grille se cierran: la región del suelo visible a través de ella lee 2.24× el valor del path trace (frente a 1.18× con todas las leaves), es decir, aparece una superficie donde debería verse el suelo.

**La causa** — `Chrome_Part_Grill` está hecha de tiras subpíxel. Dos leaves contiguas de una tira se fusionan con coste ~0: medida en su propio plano, la coverage del par no cambia. A $\tau=0.25$ y de frente, esos merges miden 3–6 px de ancho en pantalla.

**El spill $E_n$** — Sobre la misma cuadrícula del ajuste de §6.4, la fracción de lo que pinta el merge ($aG$) que no pintan sus leaves ($C$): $\sum\max(aG-C,0)/\sum aG\in[0,1]$. Es 0 si el merge queda por debajo del compuesto de sus leaves en todos los puntos y 1 si pinta donde ninguna leaf pinta. La implementación coincide con una de referencia en NumPy con correlación 0.994. Se escribe como `primvars:athenea:splat:lodSpill` y el cut puede sumarlo al error propio con peso $w_s$ y suelo $E_0$ (`--cut-spill`, `--cut-spill-floor`).

*Referencias:* código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCoverage`, buffer `spill`; `treeErrorBounded`, `spillWeight`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-spill`, `--cut-spill-floor`).
:::

| coche a 640 px, grupo 0, 0.25 px | nodos | recorte del coche | frame |
|---|---|---|---|
| $w_s=0$ | 307 665 | 16.55 dB | 22.18 dB |
| $w_s=2$ | 561 242 (+82 %) | 16.79 | 22.38 |
| $w_s=5$ | 599 983 | 16.79 | 22.38 |
| $w_s=10$ | 612 963 (+99 %) | 16.79 | 22.38 |

::: {.nota}
**Lectura de la tabla** — Grupo 0 del coche a 640 px, cut de 0.25 px. Con $w_s=2$ el número de nodos sube un 82 % (307 665 → 561 242) para ganar 0.24 dB en el recorte del coche y 0.20 dB en el frame. Subir $w_s$ a 5 o 10 añade aún más nodos (hasta +99 %) sin mejorar nada: el término ya ha saturado lo que puede arreglar.

*Referencias:* código: `apps/athenea/src/CmdDecimate.cpp` (`--cut-spill`).
:::

El spill cierra los agujeros: de 1.90 a 1.09 de las leaves en el render, y de 4.66× a 1.0 en un indicador de
agujeros en espacio de imagen. El precio es un 82–99 % más de nodos. Todo merge derrama alrededor de 0.16, y los
merges que llenan los agujeros no derraman más que los de un panel, así que ningún suelo $E_0$ separa los dos grupos.
El agujero es una propiedad de la **proyección**: el merge no cubre en 3D nada que sus vecinos no cubran, y el hueco
solo se abre donde los footprints en pantalla de las tiras se separan. Ninguna medida por nodo independiente de la
vista lo ve. El valor por defecto es $w_s=0$. El bake a 1280 px (§6.10) resuelve el problema en los datos: los
agujeros de la grille en el cut pasan de 2.24 a 1.04.

::: {.nota}
**Por qué ninguna medida por nodo lo ve** — La cuantificación: el spill cierra los agujeros (de 1.90× a 1.09× de las leaves en el render; de 4.66× a 1.0 en un indicador de agujeros en espacio de imagen), pero cada merge derrama alrededor de 0.16 y los que tapan agujeros no derraman más que los de un panel. Un suelo $E_0$ no puede separar los dos grupos.

**Una propiedad de la proyección** — En 3D, el merge de dos trozos de una tira no cubre nada que la tira y sus vecinas no cubran: el hueco entre tiras no está «dentro» de ningún nodo. El hueco aparece solo en pantalla, donde los footprints proyectados de tiras subpíxel están separados por píxeles vacíos; un merge que mide varios píxeles se extiende sobre ese hueco *en la imagen*. Lo que decide si el hueco se ve depende de la vista (dirección, distancia, y el filtro de pantalla del rasterizer, que ensancha toda Gaussian subpíxel; ver Mip-Splatting, [4]). Una medida independiente de la vista no puede capturarlo.

**La solución en los datos** — Con el bake a 1280 px (§6.9–6.10), las leaves de la grille son más finas y el cut resuelve mejor las tiras: el indicador de agujeros pasa de 2.24 a 1.04 sin spill. Por eso $w_s=0$ por defecto.

*Referencias:* [4] del §10.
:::

## 6.6 Back-face culling en closed shells

A tamaños de salida pequeños, un cut muestra las caras traseras en negro. En la grille a 393 px, el culling de caras
traseras sube las leaves sin cut de 38.2 a 41.6 dB. `--cut-cull-back` descarta un nodo cuyo cono de normales entero
mira hacia atrás:

$$\theta(n)<\tfrac{\pi}{2}\ \wedge\ n_n\cdot\frac{e-\mu_n}{\lVert e-\mu_n\rVert} < -\sin\theta(n).$$

::: {.nota}
**Caras traseras en un cut** — En un mesh cerrado, las caras que miran hacia atrás quedan tapadas por las delanteras. Con Gaussians a escala subpíxel, la cara delantera no siempre cubre del todo el píxel, y la trasera (iluminada por el interior oscuro de la pieza) añade negro. Hacer culling de las caras traseras sube las leaves de la grille a 393 px de 38.2 a 41.6 dB.

**La condición de cono** — Sea $v=(e-\mu_n)/\lVert e-\mu_n\rVert$ la dirección al ojo. Una normal mira hacia atrás si $n\cdot v<0$. Todas las normales dentro de un cono de semiángulo $\theta$ alrededor de $n_n$ miran hacia atrás si el ángulo entre $n_n$ y $v$ supera $\pi/2+\theta$, es decir, si $n_n\cdot v<\cos(\pi/2+\theta)=-\sin\theta$. La condición $\theta<\pi/2$ excluye conos que abarcan más de un hemisferio, que nunca pueden mirar enteros hacia atrás. Es la prueba de *cone of normals* para culling jerárquico de caras traseras (Shirman y Abi-Ezzi, 1993; Johannsen y Carter, 1998, *clustered backface culling*), la misma que usan los *meshlets* de las GPUs actuales.

*Referencias:* código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorKeep`, bloque «ITS BACK, NOT DRAWN»; `treeCulledMark`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-cull-back`).
:::

El cono acota cada leaf bajo el nodo (§6.3), así que todo el subárbol del nodo mira también hacia atrás. Dos medidas
fijan dónde se aplica. Aplicado a las seis piezas de prueba, con la tapa de vidrio del faro entre ellas, cuesta 6 dB a
1920: el vidrio se ve a través, así que sus caras traseras son visibles. En el coche entero a 393 px con el cut por
error de 1 px, las puertas y paneles, que son dos pieles, muestran una red de grietas oscuras: los merges de la piel
interior, mirando hacia atrás e iluminados por el interior oscuro, se ordenan por parches delante de la piel exterior.
La red es la misma con opacidad de masa o de coverage, y `--cut-guard 1000` no la cambia. En un closed shell, lo que
hay detrás de una cara que mira hacia atrás es el interior de la propia pieza, así que hacerle culling es seguro y
elimina la red. Por eso el culling se aplica, a leaves y merges por igual, solo en **closed shells que no son vidrio**
(`primvars:athenea:splat:closed`, transmisión $\le 0.5$).

::: {.nota}
**Por qué es seguro en el subárbol** — $\theta(n)$ acota la normal de cada leaf bajo $n$ (§6.3), así que si el cono mira entero hacia atrás también mira hacia atrás cada leaf: el subárbol entero puede descartarse (`treeCulledMark` lo marca para que una traversal se detenga ahí).

**Dónde no aplicarlo** — Vidrio: se ve a través, y sus caras traseras forman parte de lo visible (el culling en las seis piezas, con la tapa del faro, costó 6 dB). Superficies de dos pieles (puertas y paneles con chapa interior y exterior): sin culling, los merges de la piel interior, que miran hacia dentro e iluminados por el interior oscuro, se ordenan por parches delante de la exterior y forman una red de grietas oscuras; la red no depende de la opacidad (masa o coverage) ni de las guardas (`--cut-guard 1000`).

**Closed shell** — Una superficie cerrada (sin borde) divide el espacio en interior y exterior. Si una pieza es cerrada, lo que hay detrás de una cara que mira hacia atrás es su propio interior: descartarla no puede destapar nada que debiera verse. La marca se escribe por Gaussian como `primvars:athenea:splat:closed` y el cut la combina con transmisión $\le0.5$ (buffer `cullable`).

*Referencias:* código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeErrorKeep`, `cullable`, `treeCulledMark`), `modules/usd/src/Export.cpp` (escritura de `primvars:athenea:splat:closed`), `modules/usd/include/athenea/usd/Export.h` (`closed`, `markClosed`), `apps/athenea/src/CmdDecimate.cpp` (`--mark-closed`).
:::

**La regla de aristas abiertas.** `mesh2splat` cuenta las aristas no compartidas por exactamente dos triángulos, $O$.
Se consideraron dos reglas más simples: $O=0$, y una proporción de aristas abiertas por debajo del 2 % (111 de 192
meshes cerrados). El borde de una lámina abierta crece como la raíz cuadrada de su área, así que un tubo fino pasa la
regla de proporción: el barril de una llanta, al 1 %, recibe culling, y se ve luz entre los radios. La regla que se
usa es

$$\text{closed} \iff O < 2\sqrt{E},\qquad E = 1.5\,T,$$

para un mesh de $T$ triángulos. Con ella, las puertas (0), los logos (1.4) y las pinzas de freno (1.5) son cerrados, y
las llantas (4.0), los neumáticos (3.1), el capó (3.1) y el parachoques (4.4) no, lo que da 102 de los 192 meshes del
Corvette. Con todas las leaves a 1920 px frente al path trace, el culling sube toda la puerta un ~12 % de forma
uniforme (puerta 1.01, suelo cercano 1.02 del path trace), con 4.35 M → 3.70 M Gaussians y el recorte del coche
16.73 → 16.75 dB.

::: {.nota}
**Aristas abiertas** — En un mesh de triángulos cerrado y bien formado (*2-manifold* sin borde) cada arista la comparten exactamente dos triángulos; como cada triángulo tiene 3 aristas, $E=3T/2=1.5T$. Las aristas con 1 triángulo (borde) o con más de 2 (no manifold) cuentan como abiertas, $O$. `mesh2splat` las cuenta en un kernel (`openEdges`), soldando vértices por posición.

**Por qué $\sqrt E$** — El borde de una lámina abierta crece como la raíz de su área. Para una cuadrícula de $N\times N$ cuadrados (dos triángulos cada uno), $T=2N^2$, $E\approx3N^2$ y el borde tiene $4N=4\sqrt{E/3}\approx2.31\sqrt E$ aristas. El umbral $O<2\sqrt E$ deja fuera, por tanto, cualquier lámina abierta «cuadrada», sea cual sea su resolución. Una regla por proporción ($O/E<2\%$) falla en mallas finas: con muchos triángulos, $O/E\sim1/\sqrt E$ se hace pequeña aunque el borde sea real, y el barril de una llanta (1 %) pasaba por cerrado, con luz visible entre los radios.

**Las cifras** — Entre paréntesis, $O/\sqrt E$: puertas 0, logos 1.4, pinzas 1.5 (cerradas); llantas 4.0, neumáticos 3.1, capó 3.1, parachoques 4.4 (abiertas). 102 de 192 meshes resultan cerrados (frente a 111 con la regla del 2 %). Con todas las leaves a 1920 px, el culling sube el brillo de toda la puerta un ~12 % de forma uniforme (puerta 1.01 y suelo cercano 1.02 del path trace: ahora casi exactos), descarta 0.65 M Gaussians (4.35 M → 3.70 M) y mejora el recorte del coche 0.02 dB.

*Referencias:* código: `apps/athenea/src/CmdMesh2Splat.cpp` (bloque «A COUNTER BACK: whether the mesh is a closed shell», `closed_[h.piece] = double(open) < 2.0 * std::sqrt(meshEdges)`), `modules/usd/src/Export.cpp`.
:::

## 6.7 Occlusion masks

Un transfer TX guarda 256 bits de sombra por Gaussian, uno por cell de una cuadrícula octaédrica de $16\times16$ sobre
la esfera del mundo, activado donde un rayo desde la Gaussian llegó al cielo. Una dirección cuyas cells están todas
cerradas es una desde la que un ojo fuera del asset no puede ver la Gaussian. Los bits del shading no pueden
reutilizarse para esto: un merge guarda los bits de sus leaves por mayoría, y el codebook del player (`shadow_vq`,
§7.3) también es una mayoría, así que ambos cerrarían direcciones desde las que se ve alguna leaf.
`--occlusion-mask N` (`occlusion_mask.slang`, `Occlusion.cpp`) construye una máscara conservadora por nodo.

::: {.nota}
**Bits de sombra del transfer TX** — En §4, cada Gaussian del bake guarda 256 bits: uno por cell de una cuadrícula octaédrica $16\times16$ sobre la esfera de direcciones del mundo, a 1 donde un rayo desde la Gaussian en esa dirección escapó al cielo.

**Mapeo octaédrico** — Proyecta la esfera sobre un octaedro y lo despliega en un cuadrado (Engelhardt y Dachsbacher, 2008; Cigolle et al., 2014): parametrización casi uniforme en área, barata de codificar (`octEncode`) y con un pliegue en el borde del cuadrado que conecta el hemisferio inferior. Una cell abarca unos $180^\circ/16\approx11^\circ$.

**De la sombra a la visibilidad** — Si todas las cells de una dirección están cerradas, desde esa dirección hay un oclusor entre la Gaussian y el infinito; un ojo situado fuera del asset (más allá de todo oclusor) en esa dirección no puede ver la Gaussian. Es *occlusion culling* precalculado y dependiente de la dirección, en el espíritu de los *potentially visible sets* (Teller y Séquin, 1991) pero por elemento y por dirección.

**Por qué no reutilizar los bits del shading** — Un merge guarda los bits de sus leaves por mayoría y el codebook del player (`shadow_vq`) también: para sombrear eso es correcto, pero para descartar no, porque cerraría direcciones desde las que alguna leaf sí se ve. El culling exige máscaras **conservadoras**: abrir de más es seguro (se dibuja algo invisible), cerrar de más no (desaparece algo visible).

*Referencias:* Engelhardt, T., Dachsbacher, C. (2008) *Octahedron Environment Maps*, VMV; Cigolle, Z. et al. (2014) *A Survey of Efficient Representations for Independent Unit Vectors*, JCGT; Teller, S., Séquin, C. (1991) *Visibility Preprocessing for Interactive Walkthroughs*, SIGGRAPH; código: `shaders/athenea/lod/occlusion_mask.slang`, `shaders/athenea/lod/shadow_vq.slang`, `modules/usd/src/Occlusion.cpp`, `shaders/athenea/common/octahedral.slang`.
:::

**Máscaras.** La máscara $M_l$ de una leaf son sus cells abiertas dilatadas una cell ($3\times3$), reflejadas a
través del borde del cuadrado octaédrico en el fold. Una cell son ~11°, y sin el margen el borde de un oclusor cierra
direcciones desde las que el cromo de una grille aún se ve. La máscara está toda abierta en el vidrio (transmisión
$>0.5$, porque lo que hay detrás del vidrio se ve a través de él) y en una leaf sin bits. La máscara de un merge es el
OR de las de sus hijos, subido hasta que ninguna máscara cambia:

$$M_p = \bigvee_{k\in\mathrm{ch}(p)} M_k .$$

::: {.nota}
**Dilatación $3\times3$** — Cada cell abierta abre sus 8 vecinas (`occlusionLeaf`). Con cells de ~11°, el borde de un oclusor queda cuantizado; sin margen, una cell marcada cerrada puede incluir direcciones desde las que el cromo de la grille aún se ve por el borde. La vecindad cruza el borde del cuadrado octaédrico reflejando la cell (`across`), porque en el pliegue la vecina geométrica está al otro lado.

**Casos todo abiertos** — Vidrio (transmisión $>0.5$: lo que hay detrás se ve a través, aunque sus rayos se detuvieran en él) y leaves sin bits.

**OR hacia arriba** — $M_p=\bigvee_kM_k$: una dirección queda cerrada en un merge solo si está cerrada en todos sus hijos (y por inducción en todas sus leaves). Entonces descartar el nodo desde esa dirección es seguro para todo su subárbol. `occlusionUp` aplica `InterlockedOr` palabra a palabra y se relanza hasta que ninguna máscara cambia, como las subidas de §6.3.

*Referencias:* código: `shaders/athenea/lod/occlusion_mask.slang` (`occlusionLeaf`, `across`, `cellOpen`, `occlusionUp`).
:::

**Codebook conservador.** Las máscaras se sustituyen por un índice a $N$ entradas de 256 bits cada una. La entrada 0
está toda abierta. El **valor** de un nodo es el número de cells cerradas entre las *relevantes*, las direcciones
con $d\cdot\hat{up} > -0.25$ (un ojo hasta ~15° bajo el horizonte). Los nodos de los intervalos de menor
$\log_2$-valor, que juntos suman menos del 2 % del valor total, toman la entrada 0 y quedan fuera del ajuste. Las
demás entradas se siembran con nodos ajustados repartidos por el fichero. La ronda 1 asigna cada nodo a la entrada más
cercana en distancia de Hamming sobre las cells relevantes. Las rondas 2–8 asignan un nodo solo a una entrada que
**contiene** su conjunto abierto, eligiendo la que tiene menos cells relevantes abiertas. Tras cada ronda una entrada
pasa a ser el OR de sus miembros. Cada entrada es por tanto un superconjunto del conjunto abierto de cada miembro, y el
codebook nunca puede cerrar una dirección que un nodo tiene abierta. El coste del codebook son las cells que reabre
($M_{\text{entry}}\wedge\lnot M_n$) como fracción de todas las cells cerradas.

::: {.nota}
**Cuantización vectorial binaria** — Guardar 256 bits por nodo (32 B) para ~150 M de nodos sería caro; se sustituyen por un índice a un **codebook** de $N$ máscaras. Es *vector quantization* (Lloyd/k-means, [23]) sobre vectores binarios, donde la distancia natural es la de **Hamming** (número de bits distintos) y el centroide de un cluster no es la media, sino una máscara binaria.

**Conservador por construcción** — Ronda 1: cada nodo va a la entrada más cercana en Hamming sobre las cells relevantes (k-means binario clásico). Rondas 2–8: solo puede ir a una entrada que **contenga** su conjunto abierto ($M_n\subseteq M_{\text{entry}}$), y entre esas elige la más cerrada en cells relevantes. Tras cada ronda, cada entrada pasa a ser el OR de sus miembros (`occlusionUpdate`): por construcción, una entrada es superconjunto de cada miembro, y la sustitución nunca cierra una dirección abierta. La entrada 0 está toda abierta y siempre es válida (red de seguridad).

**Valor y relevancia** — El *valor* de un nodo es el número de cells cerradas entre las direcciones desde las que un ojo puede mirar: $d\cdot\hat{up}>-0.25$, es decir, hasta $\arcsin0.25\approx14.5^\circ$ bajo el horizonte. Un capó cerrado hacia el motor (abajo) tiene muchas cells cerradas que ninguna cámara usa. Los nodos se reparten en intervalos de $\log_2$ del valor; los intervalos más bajos, que juntos suman menos del 2 % del valor total, van a la entrada 0 y no gastan entradas. Las semillas iniciales son máscaras de nodos ajustados repartidos por el fichero.

**Coste del codebook** — Cells reabiertas, $M_{\text{entry}}\wedge\lnot M_n$, como fracción de las cells cerradas (`occlusionLost`): lo que el culling deja de descartar por la compresión.

*Referencias:* [23] del §10; Hamming, R. W. (1950) *Error Detecting and Error Correcting Codes*; código: `shaders/athenea/lod/occlusion_mask.slang` (`relevantWord`, `closedCells`, `worthBin`, `occlusionWorth`, `occlusionSeed`, `occlusionAssign`, `occlusionUpdate`, `occlusionLost`), `modules/usd/src/Occlusion.cpp` (`occlusionBook`).
:::

**Repartido por necesidad.** Los trees de un asset comparten un codebook (`--occlusion-group`). Cada unidad se
ajusta sola al tamaño completo y se cuentan las entradas que usa. Después el codebook se reparte como
$\text{share}_u = 16 + \lfloor\text{spare}\cdot \text{used}_u/\sum\text{used}\rfloor$, y cada unidad se reajusta
dentro de su parte. Repartir por calidad de ajuste y por pieza (`--occlusion-by-part`) se midió, y es peor en ambos
casos:

::: {.nota}
**Reparto por necesidad** — Un único codebook para todos los trees de un asset (`--occlusion-group`). Primero cada unidad se ajusta sola con el codebook entero, y se cuenta cuántas entradas usa de verdad, $\text{used}_u$. Luego cada unidad recibe un mínimo de 16 entradas más una parte proporcional de las sobrantes: $\text{share}_u=16+\lfloor\text{spare}\cdot\text{used}_u/\sum\text{used}\rfloor$. Cada unidad se reajusta dentro de su parte y sus índices se desplazan tras las partes anteriores. El mínimo garantiza un sitio incluso a una unidad que no descarta nada.

**Alternativas medidas** — Repartir por calidad de ajuste daba ventaja a piezas de mucho valor nominal y poco culling real; repartir por pieza (`--occlusion-by-part`, mínimo de 2 entradas por unidad) fragmenta demasiado el codebook (66 %/74 % del culling ideal frente a 84 %).

*Referencias:* código: `modules/usd/src/Occlusion.cpp` (`occlusionMasksShared`: `least = byPart ? 2u : 16u`, `share`, `spare`), `apps/athenea/src/CmdDecimate.cpp` (`--occlusion-mask`, `--occlusion-group`, `--occlusion-by-part`).
:::

| configuración | trees | codebook | reabierto | culling frente al ideal |
|---|---|---|---|---|
| un codebook por tree | 6 piezas | 256 | 3.2–10 % | — |
| un ajuste sobre todos los trees | 6 piezas | 4096 / 1024 | 3.2 % / 6.5 % | — |
| + valor, solo cells relevantes | 6 piezas | 4096 / 1024 | 21.9 % / 24.7 % | — |
| valor = cells cerradas, no leaves | 6 piezas | 4096 | 14.7 % | — |
| repartido por necesidad | 6 piezas | 4096 | 7.2 % (grille 2.8 %, 1 249 entradas) | 86 % cerca / 82 % lejos |
| 8 grupos por necesidad | coche 640 | 4096 | 16.2 % | — |
| 8 grupos por necesidad | coche 640 | 8192 | 11.9 % | 84 % |
| por pieza (192) | coche 640 | 4096 / 8192 | — | 66 % / 74 % |
| 8 grupos por necesidad | coche 1280 | 8187 | 12.3 % | — |
| 24 trees por necesidad | coche 5120 | 8182 | 0–23.6 % por tree | — |

::: {.nota}
**Lectura de la tabla** — «Reabierto»: fracción de cells cerradas que el codebook vuelve a abrir (menos es mejor). «Culling frente al ideal»: nodos descartados con el codebook respecto a los que descartarían las máscaras exactas.
- *Un codebook por tree* (256 entradas): 3.2–10 % reabierto.
- *Un ajuste sobre todos* (4096/1024): 3.2 %/6.5 %.
- *+ valor, solo cells relevantes*: el ajuste ya no intenta conservar cells bajo el horizonte; el total sobre todas las cells sube (21.9 %/24.7 %), pero no por perder culling útil (nota que sigue).
- *Valor = cells cerradas, no leaves*: ponderar por leaves favorecía los merges grandes de un guardabarros frente a las muchas barras pequeñas de la grille; un nodo del cut es una Gaussian dibujada, cubra las leaves que cubra.
- *Repartido por necesidad*: 7.2 % (grille 2.8 % con 1 249 entradas); 86 % del culling ideal de cerca y 82 % de lejos.
- *Coche a 640 px, 8 grupos*: 16.2 % con 4096, 11.9 % con 8192 (84 % del ideal). Por pieza (192 unidades): 66 %/74 %.
- *Coche a 1280 px*: 8187 entradas efectivas, 12.3 %. *Coche a 5120 px*: 24 trees, 8182 entradas, 0–23.6 % por tree. (Los números 8187/8182 en lugar de 8192 reflejan los redondeos por defecto del reparto $\lfloor\cdot\rfloor$.)

*Referencias:* código: `modules/usd/src/Occlusion.cpp`.
:::

(Desde la tercera fila, el ajuste cuenta solo las cells relevantes, así que el total reabierto sobre *todas* las cells
sube. Esos porcentajes no son comparables con los de las dos primeras filas.) En el close-up level, la suma de cells
cerradas sobre ~150 M nodos supera un contador de 32 bits, así que los totales se suman en 64 bits. Los resultados por
tree son: Windshield 0.0 % (vidrio, 1 entrada usada), Part_2 5.3 %, Hood 7.0 %, la puerta piloto 7.7 %, los grupos
finos 8.9–15.9 % y el grupo de paneles pequeños pan2 23.6 %.

::: {.nota}
**Porcentajes no comparables** — Desde la tercera fila el ajuste solo cuenta cells relevantes, así que puede dejar abiertas cells bajo el horizonte sin coste; el porcentaje sobre todas las cells sube aunque el culling útil no empeore.

**Desbordamiento de 32 bits** — ~150 M nodos × hasta 256 cells $\approx3.8\times10^{10}$, frente a $2^{32}\approx4.29\times10^9$. El primer total salió «132.4 %» porque el contador dio la vuelta; los valores por tree y el codebook eran correctos. Los totales se suman ahora en 64 bits (`uint64_t closed` en `OcclusionBook`).

**Por tree** — El parabrisas (vidrio, todo abierto) usa una entrada y reabre 0 %; las piezas grandes reabren 5–8 %, y los grupos de piezas finas y de paneles pequeños, con formas de oclusión muy variadas, hasta 23.6 %.

*Referencias:* código: `modules/usd/include/athenea/usd/Export.h` (`OcclusionBook`: `closed` en 64 bits, `seenClosed`).
:::

**La regla del player** (y `occludedFrom` en el cut). Esta regla se aplica solo cuando el ojo está fuera de la caja
envolvente del asset. Sea $d=\mathrm{normalize}(e-\mu_n)$ y $q=16\,\mathrm{oct}(d)-0.5$. Las cuatro cells
$\lfloor q\rfloor+\{0,1\}^2$, limitadas a $[0,15]^2$, son el footprint bilineal de $2\times2$ hacia el ojo. El nodo
recibe culling cuando las cuatro están cerradas en su entrada. Con el coche como oclusor (nunca en una escena
parcial), la regla quita un 12 % del cut cercano y un 8 % del lejano por 0.06 dB. En el coche entero a 393 px, con
back-face culling, la oclusión lleva el cut cercano de 522 931 a 360 333 nodos.

::: {.nota}
**La regla de lectura** — Con $d$ la dirección del nodo al ojo, $q=16\,\mathrm{oct}(d)-0.5$ sitúa la dirección en coordenadas de cell con los centros en enteros; las cuatro cells $\lfloor q\rfloor+\{0,1\}^2$ son las que entrarían en una interpolación bilineal (el *footprint* $2\times2$), limitadas a $[0,15]^2$. El nodo se descarta solo si las cuatro están cerradas: otro margen conservador además de la dilatación. Es la misma lectura que hace el shading de los bits TX (`splatCellsOpen` en `shaders/athenea/splat/splat_relight.slang`).

**Solo con el ojo fuera de la caja** — Los bits dicen si un rayo desde la Gaussian llega al infinito. Un ojo dentro de la caja envolvente podría estar entre la Gaussian y su oclusor, y vería algo que los bits declaran ocluido. Y solo con el asset completo como oclusor: en una escena parcial (unas pocas piezas) los oclusores que cerraron las direcciones no están.

**Cifras** — 12 % menos nodos en el cut cercano y 8 % en el lejano por 0.06 dB. En el coche entero a 393 px con back-face culling, el cut cercano baja de 522 931 a 360 333 nodos (−31 %).

*Referencias:* código: `shaders/athenea/lod/lod_tree_cut.slang` (`occludedFrom`, `treeErrorKeep`, `treeCulledMark`), `shaders/athenea/common/octahedral.slang` (`octEncode`), `shaders/athenea/splat/splat_relight.slang` (`splatCellsOpen`), `apps/athenea/src/CmdDecimate.cpp` (`--cut-occlusion`).
:::

## 6.8 Pages por error

Un player no puede tener un tree entero. Necesita unidades que traiga enteras y que pruebe sin leer sus nodos.

::: {.nota}
**Pages** — Unidades de streaming de tamaño fijo (aquí $P$ nodos). El player las trae enteras y decide si las necesita mirando solo una cabecera (caja y errores), sin leer sus nodos: así puede gestionar la residencia con un coste proporcional al número de pages, no de nodos. Es el papel de las *pages* de Nanite ([35]) o de los *tiles* de 3D Tiles.

*Referencias:* [35] del §10; código: `shaders/athenea/lod/lod_pages.slang`.
:::

**Alternativa considerada: pages por nivel.** Las pages tomadas como cimas en anchura de subárboles hermanos siguen
necesitando 114 de las 263 pages del capó para un cut lejano. Las cotas conservan nodos pequeños dondequiera que gira
un brillo o un color, así que el cut abarca nodos desde leaves sueltas hasta $2^{11}$ leaves.

::: {.nota}
**Pages por nivel (descartadas)** — Tomar como page la cima de un subárbol, recorrido en anchura, y agrupar hermanos: es la paginación natural de una jerarquía. Falla porque un cut del merge tree no está a un nivel: las cotas de color y normal (§6.3) mantienen nodos pequeños dondequiera que un brillo gira o cambia un color, y el cut mezcla leaves sueltas con nodos de $2^{11}=2048$ leaves. Una page de subárbol contiene nodos de muchos tamaños, y un cut lejano del capó toca 114 de sus 263 pages.

*Referencias:* código: `shaders/athenea/lod/lod_pages.slang` (`treePagesTouched`, `treePagesCount`: la paginación por nivel de `--cut-pages`).
:::

**Pages por error.** Un nodo se necesita exactamente cuando es una raíz o el error proyectado de su padre supera
$\tau$. Así que la disposición sigue el error del padre (`lod_pages.slang`). La **banda** de un nodo es

$$b(n) = \mathrm{clamp}\big(\lfloor\log_2\varepsilon(\mathrm{par}(n))\rfloor+40,\ 0,\ 63\big),\qquad b(\text{root}) = 63 .$$

::: {.nota}
**Cuándo se necesita un nodo** — Una traversal de arriba abajo lee un nodo para decidir si lo dibuja o baja a sus hijos. Lee un nodo exactamente cuando su padre se refina, es decir, cuando $P(\mathrm{par}(n))>\tau$ (o es raíz). Por eso la paginación sigue **el error del padre**, no el propio.

**La banda** — $b(n)=\lfloor\log_2\varepsilon(\mathrm{par}(n))\rfloor+40$, limitada a $[0,63]$: la octava del error del padre en metros. El desplazamiento 40 coloca la banda 0 en $2^{-40}\approx9\times10^{-13}$ m y la 63 en $2^{23}$ m: todo error útil cae dentro. Una raíz va a la banda 63 (siempre necesaria). Para un $\tau$ y una distancia dados, se necesitan las bandas con $\varepsilon f/d>\tau$: un prefijo de las bandas ordenadas de más gruesa a más fina.

*Referencias:* [35] del §10; código: `shaders/athenea/lod/lod_pages.slang` (`bandOf`, comentario «PAGES BY ERROR (Nanite's order)»).
:::

La clave de sort, en el orden propio de las pages, es (banda más gruesa primero, después primera leaf, después
el nodo más grande primero):

$$\text{hi} = (63-b)\ll22 \;|\; \text{bucket}\ll19 \;|\; \text{start}\gg6,\qquad \text{lo} = (\text{start}\,\&\,63)\ll26 \;|\; (2^{26}-1-\text{leaves}).$$

::: {.nota}
**La clave de 64 bits** — Dos palabras de 32 bits que un radix sort en la GPU ordena como una sola:
- `hi`: bits 22–27 $=63-b$ (6 bits, banda más gruesa primero); bits 19–21 = cubeta de orientación (3 bits, 0 salvo con `--pages-by-facing`); bits 0–18 = `start >> 6` (los 19 bits altos de la primera leaf).
- `lo`: bits 26–31 = `start & 63` (los 6 bits bajos de la primera leaf); bits 0–25 = $2^{26}-1-\text{leaves}$ (más leaves → clave menor → antes).

Así, dentro de una banda los nodos quedan en el orden espacial de sus leaves (orden de profundidad = localidad), y a igual primera leaf, el más grande antes. `start` admite hasta $2^{25}$ leaves (`min(runStart, 0x1FFFFFF)`).

*Referencias:* código: `shaders/athenea/lod/lod_pages.slang` (`treeBandKeys`), `modules/gpu/include/athenea/gpu/algo/RadixSort.h`.
:::

Un sort radix en la GPU ordena los nodos, y la page $p$ contiene las posiciones ordenadas $[pP,(p+1)P)$. Las
pages cruzan los límites de banda, para que una banda gruesa de pocos nodos no sea una page por sí sola. La banda de
un padre nunca es más fina que la de su hijo, su primera leaf nunca es posterior, y cuando ambas empatan el padre es
el nodo más grande. Por tanto un padre nunca se coloca después de su hijo. (Una disposición que recorre al revés las
leaves de cada banda pone 92 padres después de sus hijos en el capó.) `--tree-pages P` escribe, para cada nodo,
`lodPage`, y para cada page:

- `lodPageInfo` = (banda más gruesa, nodos, primer nodo en orden de page), y
- `lodPageBounds` = la caja que contiene las esferas de los **padres** (la necesidad de un nodo se mide desde su
  padre) y el error de padre mayor y menor.

::: {.nota}
**Padre antes que hijo** — Para que una traversal pueda empezar por las pages gruesas sin esperar a las finas, un padre nunca debe quedar detrás de su hijo en el orden. Se cumple porque, comparando claves: (i) la banda del padre nunca es más fina (el error del abuelo $\ge$ el del padre, por monotonía); (ii) si empatan, la primera leaf del padre nunca es posterior (el tramo del padre contiene al del hijo); (iii) si también empatan, el padre tiene más leaves y gana el desempate. Recorrer las leaves al revés dentro de una banda rompía (ii): 92 padres detrás de sus hijos en el capó.

**Pages que cruzan bandas** — La page $p$ son las posiciones $[pP,(p+1)P)$ del orden total, sin alinear a bandas: una banda gruesa con pocos nodos no ocupa una page casi vacía.

**Tabla de pages** — `lodPageInfo` (banda más gruesa, número de nodos, primer nodo) y `lodPageBounds`: caja que contiene las esferas de los **padres** de sus nodos (la necesidad se mide desde el padre), y el mayor y el menor error de padre. Las cajas se acumulan con `InterlockedMin`/`InterlockedMax` sobre floats convertidos a un entero ordenable (`ordered`: invierte los negativos y marca el bit de signo de los positivos).

*Referencias:* código: `shaders/athenea/lod/lod_pages.slang` (`treeBandKeys`, `treeBandTable`, `treeBandTableOut`, `ordered`, `unordered`), `modules/usd/src/Decimate.cpp` (escritura de `lodPage`, `lodPageInfo`, `lodPageBounds`), `apps/athenea/src/CmdDecimate.cpp` (`--tree-pages`).
:::

El player proyecta el error mayor desde la caja y carga la page cuando esa proyección supera $\tau$. Los bits de
cubeta (`--pages-by-facing`: eje dominante de la normal y su signo) empeoran la residencia, 4.45× frente a 3.98× con
$\tau=2.5$, y están desactivados.

::: {.nota}
**Prueba del player** — Proyecta el error de padre **mayor** de la page desde el punto más cercano de su caja; si supera $\tau$, algún nodo de la page podría necesitarse y la page se carga. Es una cota conservadora: nunca descarta una page necesaria, a costa de cargar algunas que no lo son.

**Cubetas por orientación** — Agrupar dentro de cada banda por eje dominante de la normal y su signo (6 cubetas) pretendía que el back-face culling descartara pages enteras. Empeora la residencia (4.45× frente a 3.98× el cut, con $\tau=2.5$): partir por orientación rompe la localidad espacial, y cada vista sigue necesitando una parte de casi todas las cubetas. Desactivado.

*Referencias:* código: `shaders/athenea/lod/lod_pages.slang` (`treeBandKeys`, `facingBuckets`), `apps/athenea/src/CmdDecimate.cpp` (`--pages-by-facing`).
:::

**Residencia.** Una traversal necesita el cut y todos sus antepasados, alrededor de 2× el cut. Las pages contienen
alrededor de 4× el cut sea cual sea su tamaño u orden, porque los nodos necesarios y los descartados comparten cada
page mientras que el culling depende de la vista.

| configuración | escena | $\tau$ | cut | nodos necesarios | pages contienen (× cut) |
|---|---|---|---|---|---|
| P 3000 | seis piezas, cerca 1920 | 1 | 244 114 | — | 2.0–2.6× |
| P 3000 | seis piezas, lejos 393 | — | — | — | 2.2–4.9× (capó: 2 pages) |
| P 3000 | coche 640, cerca 393 | 2 | 293 629 | 647 707 (2.20×) | 1 170 000 (3.98×) |
| P 3000 | coche 640, cerca 393 | 2.5 | 263 918 | 580 997 (2.20×) | 1 050 000 (3.97×) |
| P 3000 | coche 640, cerca 393 | 3 | 198 702 | 445 195 (2.24×) | 957 000 (4.81×) |
| P 3000 | coche 640, cerca 393 | 6.75 | 92 987 | 214 827 (2.31×) | 423 000 (4.54×) |
| P 1000 / 500 | coche 640, cerca 393 | 2 | 293 629 | — | 3.91× / 3.87× |
| P 1000 / 500 | coche 640, cerca 393 | 2.5 | 263 918 | — | 3.77× / 3.60× |

::: {.nota}
**Por qué ~2× el cut** — Una traversal necesita el cut y todos sus antepasados. En un árbol binario, los antepasados de un conjunto de nodos son a lo sumo tantos como el conjunto (cada nodo interno del «árbol superior» tiene dos hijos), así que el total ronda $2\times$: 2.20–2.31× en la tabla.

**Por qué ~4× en pages** — Las pages agrupan por banda de error y por posición, pero el culling (caras traseras, oclusión) depende de la vista. Cada page mezcla nodos necesarios y nodos que esta vista descarta, y hay que traerla entera. El factor ronda 4× sea cual sea el tamaño ($P$ 3000, 1000, 500) o el orden.

**Lectura de la tabla** — Coche a 640 px, vista cercana a 393 px: con $\tau=2$ el cut tiene 293 629 nodos, la traversal necesita 647 707 (2.20×) y las pages que los contienen suman 1.17 M (3.98×). Con $\tau=6.75$ el cut baja a 92 987 y las pages a 423 k (4.54×). Pages más pequeñas mejoran poco (3.91×/3.87× con $P=1000/500$ a $\tau=2$). En las seis piezas, 2.0–2.6× de cerca y 2.2–4.9× de lejos (el capó entero en 2 pages).

*Referencias:* código: `shaders/athenea/lod/lod_pages.slang` (`treeBandPages`, `treeErrorPagesCount`, `reachable`).
:::

`treeCulledMark` marca los nodos que los culls dejan fuera. El subárbol de un nodo descartado también queda
descartado, así que una traversal se detiene ahí. Eso deja los nodos necesarios en 2.2× (una traversal dentro de
subárboles descartados necesita ~4×), pero las pages siguen conteniendo 3.6–4.5×. Un player con streaming puede por
tanto guardar la traversal a 12 B por nodo para lo que se necesita y la carga útil en tramos deflate de 64 para lo
que se dibuja; en el player WebGPU de §7, de cerca con $\tau=3$, eso supone ~60 MB y 226 k dibujadas a 60 fps
(emulado).

::: {.nota}
**`treeCulledMark`** — Marca los nodos que los culls (cara trasera en closed shell, oclusión) dejan fuera desde la vista. Como el cono y la máscara acotan todo el subárbol, una traversal se detiene en el nodo marcado; `reachable` en `lod_pages.slang` comprueba que ningún antepasado esté marcado al contar los nodos necesarios. Eso mantiene los necesarios en 2.2× (sin la marca, una traversal que entra en subárboles descartados necesita ~4×), pero las pages siguen en 3.6–4.5×: la paginación es independiente de la vista.

**Separar traversal y carga útil** — La traversal solo necesita, por nodo, lo que decide: centro, radio, error, enlaces (12 B por nodo en la cifra del texto, con cuantización). La carga útil (forma, color, transfer) solo hace falta para lo que se dibuja y puede ir en tramos comprimidos con deflate (RFC 1951) de 64 nodos. Así la residencia de la parte pesada sigue al cut, no a las pages. En el player WebGPU de §7, de cerca con $\tau=3$: ~60 MB y 226 k Gaussians dibujadas a 60 fps (en un teléfono emulado).

*Referencias:* [46] del §10; Deutsch, P. (1996) RFC 1951 *DEFLATE Compressed Data Format*; código: `shaders/athenea/lod/lod_tree_cut.slang` (`treeCulledMark`), `shaders/athenea/lod/lod_pages.slang` (`reachable`). El player es sparkwebGPU (Apéndice A.1; demos en https://sparkwebgpu.lucab.co.uk) y no forma parte del repositorio del motor.
:::

## 6.9 El close-up level

De cerca, el logo y la grille siguen blandos con el bake a 1280 px. Una cámara cercana llega a unos 0.5 mm por píxel
de salida, mientras que las leaves a 1280 px miden 3–4 mm. Un bake de logo y grille desde una cámara cercana a
5120 px muestra nítidos la bandera a cuadros, la pajarita y la malla metálica de la grille (108 527 → 404 594
Gaussians). El close-up level (car5120) vuelve a hacer el bake del coche por grupos (figura 10):

- **Piezas detalladas.** Las 66 piezas a las que el bake a 1280 px da como mucho 1.5 Gaussians por triángulo
  (emblemas, grilles, faros, pinzas, retrovisores, ruedas, limpiaparabrisas, molduras), a 5120 px, en 7 grupos
  (`fine0–6`).
- **Paneles.** Cada pieza a 5120 px, o tan fina como la mantenga bajo un tope de 4.5 M leaves (un tree de 5.01 M
  leaves está en el límite de la memoria del device al escribirse), con las pequeñas agrupadas hasta 4 M. Hace
  falta `--max-cells 1048576`: a 4736 px, Infinite_Black pierde 359 628 cells más allá de las 262 144 por triángulo
  por defecto, lo que deja agujeros.

::: {.nota}
**Por qué un close-up level** — A 1280 px de ancho de la cámara de conversión, las leaves miden 3–4 mm en el capó; una cámara cercana en el teléfono llega a ~0.5 mm por píxel. Un factor 6–8 en tamaño requiere ~4× más resolución lineal: 5120 = 4 × 1280, y ~16× más Gaussians donde la geometría lo permite. La prueba de logo y grille pasó de 108 527 a 404 594 Gaussians (3.7×) y mostró nítidos la bandera a cuadros, la pajarita (el emblema de Chevrolet) y la malla.

**Piezas detalladas** — Las 66 piezas con como mucho 1.5 Gaussians por triángulo a 1280 px están limitadas por la teselación, no por la resolución: a 1280 px ya caben un par de Gaussians por triángulo, y por eso son las que más ganan con 5120 px (cada triángulo se subdivide en más cells).

**Paneles y el tope de 4.5 M** — Un tree de 5.01 M leaves está en el límite de memoria del device al escribirse (§6.9, más abajo). Cada panel se convierte a 5120 px o a la resolución más alta que lo mantiene bajo el tope (de ahí 3392, 4480, 4736 px en la tabla).

**`--max-cells`** — `mesh2splat` recorre como mucho `maxCells` cells por triángulo (por defecto $2^{18}=262\,144$, `CmdMesh2Splat.cpp`); el resto no se muestrea y avisa con un warning. Infinite_Black, con triángulos muy grandes, perdía 359 628 cells (agujeros) a 4736 px; con $2^{20}=1\,048\,576$ se rehízo entero.

*Referencias:* [12], [13] del §10; código: `apps/athenea/src/CmdMesh2Splat.cpp` (`maxCells = 1u << 18`, aviso «cells lay past --max-cells»).
:::

Cada grupo se convierte con todas las demás piezas ocultas, pero el bake traza el escenario entero, así que una pieza
conserva las sombras de las otras. Los trees son por pieza, así que el close-up level tiene el mismo formato que
car1280: las raíces gruesas se quedan, y los niveles más profundos solo se paginan de cerca.

::: {.nota}
**Conversión aislada, bake global** — `--hide` solo afecta a la conversión: el bake traza el escenario entero, así que cada pieza conserva las sombras e interreflexiones de las demás aunque se convierta sola. Es lo que permite partir en grupos sin artefactos de sombra.

**Mismo formato que car1280** — Como los trees son por pieza (la guarda de clave), el close-up level es un conjunto más de trees con el mismo esquema: sus raíces gruesas se quedan residentes y los niveles finos solo se paginan cuando la cámara se acerca (por error proyectado, §6.8).

*Referencias:* código: `apps/athenea/src/CmdMesh2Splat.cpp` (`--hide`).
:::

| grupos | leaves | resolución |
|---|---|---|
| fine0–6 (66 piezas detalladas) | 18 610 627 | 5120 |
| puerta (piloto) | 4 084 170 | 5120 |
| Bottom | 4 897 428 | 3392 |
| Part_3 | 4 370 459 | 4480 |
| Infinite_Black | 4 481 738 | 4736 |
| Panel_under_Door_001, Windshield, Front_Fender, Hood, Front_Bumper, Part_2, Part_23 | 15 994 566 | 5120 |
| pan0–4 (paneles pequeños) | 17 088 152 | 5120 |
| Front_Bumper_001 | 5 008 986 | 4480 |
| **24 trees** | **74 536 126 leaves, 74 328 031 merges, 208 095 raíces** | |

::: {.nota}
**Lectura de la tabla** — 24 trees: 7 de piezas finas (18.6 M), la puerta del piloto, Bottom, Part_3, Infinite_Black, 7 piezas grandes a 5120 px (16.0 M), 5 grupos de paneles pequeños (17.1 M) y Front_Bumper_001. Las leaves suman 74 536 126.

**Comprobación** — En un bosque binario, $\text{raíces}=\text{leaves}-\text{merges}=74\,536\,126-74\,328\,031=208\,095$, como dice la tabla. Los nodos son $74\,536\,126+74\,328\,031=148\,864\,157$ (párrafo siguiente).
:::

Los 24 trees contienen 148 864 157 nodos en 49 634 pages de 3000, y 87.4 GB en disco. Rehechos a partir de sus
leaves con los segundos momentos respecto al propio centro de cada grupo (§6.2.1), y con un codebook de oclusión
compartido de 8181 entradas (el 11.9 % de las celdas reabiertas), el cut del coche a 0.25 px dibuja 11.7 M de Gaussians
en lugar de 16.9 M y da 24.00 dB en el frame y 18.31 dB en el coche frente al path trace (antes 23.62 y 17.91). Algunos trees conservan
muchas raíces sueltas (fine5: 29 213; Infinite_Black: 92 773), porque los merges nunca cruzan piezas ni guardas, por
diseño. El empaquetador del player (§7.5.4) añade encima una raíz sintética de varios niveles, ya que un fichero
contiene unas 3 000 raíces.

::: {.nota}
**Pages y disco** — $148\,864\,157/3000\approx49\,621$ pages si todo fuera un único orden; con 24 trees y la última page de cada uno incompleta salen algunas más, 49 634. 87.4 GB / 148.9 M nodos ≈ 590 B por nodo: el transfer TX, los bits de sombra y los demás atributos dominan sobre la geometría.

**Rehecho tras el arreglo de §6.2.1** — Con los segundos momentos centrados y el codebook compartido de 8181 entradas (11.9 % reabierto), el cut a 0.25 px dibuja 11.7 M en lugar de 16.9 M (−31 %) y mejora +0.38 dB en el frame y +0.40 dB en el coche frente al path trace.

**Raíces sueltas** — fine5 conserva 29 213 raíces e Infinite_Black 92 773: las guardas (clave, vidrio, espejo) siguen activas incluso en la cima relajada, por diseño. Como un fichero contiene unas 3 000 raíces, el empaquetador del player (§7.5.4) añade una raíz sintética de varios niveles encima, para que la traversal empiece en un solo nodo.

*Referencias:* `docs/decisions.md` («A merge's second moment about its own centre (13 Oct)»: «the 5120 px trees are to be rebuilt from their bakes»).
:::

**Memoria para trees de 4–5 M leaves.** Una rueda delantera con 4.45 M leaves y 8.88 M nodos (solo el neumático
2.56 M) construye su tree dentro de la memoria del device, pero escribirlo requiere dos cambios.

1. `mergeAttribute` necesita un buffer de pesos solo para los joints de un rig. Reservado tan grande como cada
   atributo, cuesta 1.7 GB junto a cada uno de los arrays de 48 floats de un transfer sobre 8.9 M nodos. Es una
   palabra para cualquier otro atributo.
2. La escritura no debe tener cada array tres veces en memoria unificada (lo leído, una copia, y los arrays
   fusionados empezados como otra copia). Los arrays fusionados empiezan solo desde los metadatos, y la escritura del
   tree, el último paso, libera cada array en cuanto está en el device.

Con estos cambios, una puerta de 4.08 M leaves tiene un pico de 10.5 GB residentes y se escriben 4.90 M leaves.
Front_Bumper_001 (5.01 M) se escribe con 2.23 GB de memoria libre en el punto más bajo; el tope de 4.5 M para los
paneles mantiene ese margen.

::: {.nota}
**1. El buffer de pesos** — `Decimator::mergeAttribute` (`modules/lod/src/Lod.cpp`) fusiona un atributo por tramos de leaves. Solo el modo de joints (mode 3, los joints de un rig) escribe pesos junto a los valores. Antes se reservaba un buffer de pesos del tamaño de cada atributo: $48\ \text{floats}\times8.9\,\text{M nodos}\times4\,\text{B}\approx1.7$ GB por cada array de transfer. Ahora es una sola palabra salvo para joints.

**2. Tres copias en memoria unificada** — En Apple Silicon la GPU y la CPU comparten la memoria física: lo que el host retiene compite con el device. La escritura retenía lo leído, una copia, y los arrays fusionados empezados como otra copia. Ahora los arrays fusionados empiezan desde los metadatos, y la escritura libera cada array en cuanto está en el device.

**Resultados** — Una puerta de 4.08 M leaves llega a un pico de 10.5 GB residentes. Front_Bumper_001 (5.01 M) se escribe con 2.23 GB libres en el punto más bajo; el tope de 4.5 M leaves por panel mantiene ese margen.

*Referencias:* código: `modules/lod/src/Lod.cpp` (`Decimator::mergeAttribute`, comentario «Only a rig's joints (mode 3) write weights»), `modules/lod/include/athenea/lod/Lod.h` (`mergeAttribute`), `modules/usd/src/Decimate.cpp`.
:::

**Oclusión escrita en el sitio.** Con 87.4 GB para los 24 trees, una copia de cada tree no cabe junto a ellos en
disco. `occlusionMasksShared` escribe cada tree en `<tree>.occlusion.usdc` a su lado y lo renombra sobre el original.
El codebook compartido (8 182 entradas) tarda 645 s, con un pico de conjunto residente de 10.2 GB; por tree reabre
del 0 al 23.6 % de las cells cerradas (§6.7).

::: {.nota}
**Escritura en el sitio** — `occlusionMasksShared` escribe cada tree con sus máscaras a `<tree>.occlusion.usdc` y lo renombra sobre el original. El renombrado dentro de un mismo sistema de ficheros es atómico (POSIX `rename`): en ningún momento coexisten dos copias completas de los 87.4 GB, y un fallo deja el original intacto.

**Coste** — 645 s y 10.2 GB de pico para un codebook compartido de 8 182 entradas sobre los 24 trees; cada tree reabre del 0 al 23.6 % de sus cells cerradas (§6.7).

*Referencias:* código: `modules/usd/src/Occlusion.cpp` (`occlusionMasksShared`), `modules/usd/include/athenea/usd/Export.h`, `apps/athenea/src/CmdDecimate.cpp`.
:::

## 6.10 Resultados medidos

Todos los renders son del rasterizer del motor. «Leaves a 4×» significa todas las leaves dibujadas a 4× de resolución
y reducidas por caja, la referencia para cuts lejanos (un bake a 1920 px dibujado a 393 px se transparenta). «PT» es
el path trace del mesh a 1920×1080 desde `/World/Camera`, comparado en el frame y en el recorte de 900×420 del coche.

::: {.nota}
**«Leaves a 4×»** — Todas las leaves dibujadas a 4× la resolución de salida y reducidas con un filtro de caja ($4\times4$ píxeles → 1): un render supermuestreado, la referencia para cuts lejanos. Hace falta porque un bake dimensionado para 1920 px, dibujado a 393 px, deja cada Gaussian muy por debajo del píxel y la imagen «se transparenta» (el problema de aliasing y escala de 3DGS que trata Mip-Splatting, [4]). Comparar un cut contra las leaves a resolución nativa mediría ese defecto, no el del LOD.

**«PT»** — El path trace del mesh original (§2), 1920×1080 desde `/World/Camera`: la verdad de referencia de todo el pipeline. Se mide en el frame completo y en un recorte de 900×420 que encuadra el coche.

*Referencias:* [4] del §10; código: `apps/athenea/src/CmdCompare.cpp` (`athenea compare`); `docs/operations.md` §2.11 «`athenea compare` — what an image holds, measured on the GPU».
:::

**El coche a 640 px a 393 px, frente a las leaves a 4×.**

| configuración | cut | nodos cerca | dB cerca | nodos lejos | dB lejos |
|---|---|---|---|---|---|
| trees con coverage | error 1 px, cotas | 666 585 | 33.29 | 343 541 | 39.15 |
| trees con masa | error 1 px, cotas | 666 585 | 33.27 | — | — |
| trees con coverage | $\tau=0.5$ | 930 119 | 33.17 | — | — |
| + back-face culling | closed shells por la regla del 2 % | 522 931 | 33.63 | — | — |
| + oclusión | 8192 repartidas por grupo | 360 333 | 33.58 | 195 049 | 39.53 |
| ambos culls | regla $O<2\sqrt{E}$ (102 piezas) | 378 028 | 33.53 | — | — |

::: {.nota}
**Lectura de la tabla** (coche a 640 px, salida 393 px, frente a las leaves a 4×):
- *Trees con coverage, error 1 px con cotas*: 666 585 nodos de cerca a 33.29 dB; 343 541 de lejos a 39.15 dB.
- *Trees con masa*: mismos nodos, 33.27 dB: la opacidad apenas importa a esta escala.
- *$\tau=0.5$*: 930 119 nodos a 33.17 dB: bajar el umbral sin cotas añade nodos sin ganar calidad.
- *+ back-face culling* (regla del 2 %): 522 931 nodos y +0.34 dB (desaparece la red de grietas).
- *+ oclusión* (8192 entradas repartidas por grupo): 360 333 de cerca (−31 %) a 33.58 dB y 195 049 de lejos a 39.53 dB.
- *Ambos culls con la regla $O<2\sqrt E$*: 378 028 nodos a 33.53 dB; algo más que con la regla del 2 % porque menos piezas (102 frente a 111) son closed shells, y se corrige la luz entre los radios.

*Referencias:* código: `shaders/athenea/lod/lod_tree_cut.slang`.
:::

El cut lejano (195 k) cabe en un budget de ~250 k nodos, el que el player móvil de §7 dibuja a 30 fps. El cut
cercano es ~1.4× eso.

::: {.nota}
**Budget del player móvil** — ~250 k nodos es lo que el player de §7 dibuja a 30 fps en el teléfono de referencia. El cut lejano (195 k) cabe; el cercano (360 k) lo supera en ~1.4×. El player fija su objetivo de dibujo por tiempo de frame, así que de cerca dibuja un cut algo más grueso.

*Referencias:* [11], [46] del §10.
:::

**Frente al path trace (1920 px).**

| bake | qué | nodos | recorte del coche | frame |
|---|---|---|---|---|
| 640 px | todas las leaves | 5.38 M | 16.37 | 21.92 |
| 640 px | error 1 px, ambos culls | 675 760 | 16.28 (31.91 frente a leaves) | 21.80 |
| 640 px | error 0.25 px, back-face culling | 2 313 541 | 16.29 (34.01 frente a leaves) | 21.81 |
| 640 px | leaves, sharp lobes sin ensanchar | 5.38 M | 16.58 | 22.15 |
| 640 px | 0.25 px, back-face culling, correcciones de shading | — | 16.55 | 22.18 |
| 640 px | leaves (0.01 px), sin cull / con cull | 4.35 M / 3.70 M | 16.73 / 16.75 | 22.40 / 22.43 |
| 640 px | grupo 0 con spill $w_s=2$ | 561 242 (g0) | 16.79 | 22.38 |
| **1280 px** | 0.25 px, back-face culling | 4 998 976 | **17.88** | **23.90** |
| 1280 px | 0.25 px, back-face culling, separación tubo/fold (§4.9.4) | — | 17.88 | 23.91 |

::: {.nota}
**Lectura de la tabla** (frente al path trace a 1920 px; PSNR del recorte del coche y del frame):
- *Todas las leaves a 640 px*: 5.38 M Gaussians, 16.37 / 21.92 dB: el techo que el LOD puede alcanzar con ese bake.
- *Error 1 px, ambos culls*: 675 760 nodos (1/8 de las leaves), 16.28 dB, solo 0.09 dB menos; frente a las propias leaves, 31.91 dB.
- *Error 0.25 px, back-face culling*: 2.31 M nodos, 16.29 dB (34.01 frente a leaves).
- *Sharp lobes sin ensanchar* y *correcciones de shading* (§4): mejoras del bake/shading, no del LOD (16.58, 16.55).
- *Leaves (cut a 0.01 px), sin/con cull*: 4.35 M / 3.70 M; 16.73 / 16.75 dB. El cut a 0.01 px es prácticamente el conjunto de leaves; que el recuento quede por debajo de 5.38 M indica que algunos merges (por ejemplo, de Gaussians casi coincidentes) tienen error proyectado por debajo de 0.01 px y sustituyen a sus leaves.
- *Grupo 0 con spill* (§6.5): 16.79 dB.
- *1280 px, 0.25 px con back-face culling*: 5.0 M nodos, **17.88 / 23.90** dB: el salto lo da el bake.
- *Separación tubo/fold* (§4.9.4): +0.01 dB en el frame.

**PSNR bajas en absoluto** — 16–24 dB frente al path trace no son errores del LOD: comparan dos renderers distintos (Gaussians horneadas frente a un path trace del mesh) e incluyen diferencias de brillos, faros y nitidez de reflejos. Lo que mide el LOD es la diferencia entre filas con el mismo bake.
:::

El LOD cuesta alrededor de 0.1 dB frente al path trace (16.37 → 16.28 con 1/8 de las leaves). El bake explica el
resto: su resolución, sus faros y la nitidez de sus reflejos. El nuevo bake a 1280 px (8.69 M leaves, 8.67 M merges,
20 723 raíces, 5 789 pages, la oclusión reabriendo un 12.3 %) gana 1.3 dB en el coche. Frente al cut de 0.25 px a
640 px con las correcciones de shading, el cut a 1280 px lleva los agujeros de la grille de 2.24 a 1.04 del path
trace, los brillos del capó de 3.2 a 2.0 px de desvío, y las juntas de 0.90 a 1.02 del path trace.

::: {.nota}
**El coste del LOD** — 16.37 → 16.28 dB con 675 760 nodos de 5.38 M (12.6 %, ~1/8): unos 0.1 dB. El resto del error frente al path trace es del bake: resolución, faros (las luces de los faros no estaban en el bake) y nitidez de los reflejos.

**El bake a 1280 px** — 8.69 M leaves, 8.67 M merges, 20 723 raíces ($8.69\,\text{M}-8.67\,\text{M}\approx0.02$ M, consistente), 5 789 pages de 3000 (≈17.36 M nodos / 3000) y 12.3 % reabierto en oclusión. Gana +1.33 dB en el coche (16.55 → 17.88). Agujeros de la grille: de 2.24 a 1.04 del path trace (1.0 sería exacto). Brillos del capó: de 3.2 a 2.0 px de desvío de posición. Juntas: de 0.90 a 1.02 del path trace (antes demasiado oscuras, ahora casi exactas).
:::

## 6.11 Límites y cuestiones abiertas

- **Agujeros solo de proyección.** Un cut grueso de tiras subpíxel (la grille) llena agujeros que ninguna medida por
  nodo e independiente de la vista puede ver (§6.5). En la práctica se resuelve con leaves más finas, no con el cut.
- **Ponderación de la oclusión.** El codebook se reparte por entradas usadas, no por la frecuencia con que una vista
  descarta un nodo. Ponderar por una órbita de vistas en el bake debería subir el culling por encima del 84 % del
  ideal.
- **Residencia en ~4×.** La limitan pages que mezclan nodos necesarios y nodos descartados por la vista. Un player con
  streaming separa la traversal de la carga útil en vez de encoger las pages.
- **Merges de vidrio.** Cada Gaussian de lámina fina se eleva al suelo de lámina de 0.1, así que fusionar en menos
  láminas más grandes hace variar la transmisión de cell a cell. Mientras tanto Spark declara los merges de lámina a
  4× su tamaño, y queda pendiente un suelo por píxel en lugar de por Gaussian.
- **Las molduras finas y lejanas** siguen convirtiéndose en trazos gruesos en todos los cuts. Las cotas no ven ese
  error.
- **Sin portar.** La regla de caras traseras en closed shells y el cut por error viven en una rama de desarrollo. El
  `CutSelector` del motor principal no ha cambiado.

::: {.nota}
**Agujeros solo de proyección** — Véase §6.5: un defecto que solo existe en pantalla no puede medirlo un error independiente de la vista. Las salidas posibles son leaves más finas (lo adoptado) o una medida dependiente de la vista calculada en el propio cut.

**Ponderación de la oclusión** — El reparto del codebook usa entradas usadas como medida de necesidad. Lo que importa es cuántas veces una vista real descarta un nodo; ponderar por una órbita de cámaras durante el bake aproximaría esa frecuencia (como hace `--drop-hidden` para la visibilidad, `lod_prune.slang`).

**Residencia en ~4×** — Encoger pages no la reduce (§6.8: 3.6–3.9× con $P=500$). La salida es separar la traversal, compacta, de la carga útil, que se trae por tramos según lo dibujado.

**Merges de vidrio** — El vidrio fino se dibuja como lámina con opacidad mínima 0.1 (`kSheetAlpha` en `shaders/athenea/splat/splat_project.slang`). Al fusionar láminas en menos y más grandes, ese suelo por Gaussian cambia la transmisión total de una zona a otra. Spark compensa declarando los merges de lámina a 4× su tamaño; un suelo por píxel en lugar de por Gaussian es lo pendiente.

**Molduras finas y lejanas** — Una moldura de cromo subpíxel se convierte en un trazo grueso: las cotas de color y normal no ven el error de anchura de un objeto lineal fino (el mismo problema de forma que la guarda de aguja, desactivada por defecto).

**Sin portar** — En el repositorio del motor, el cut por error con cotas (`treeErrorKeep`, `treeBoundsUp`), el back-face culling de closed shells (`--cut-cull-back`, buffer `cullable`), la oclusión y las pages existen en la herramienta offline `athenea decimate` (`modules/usd/src/Decimate.cpp`, `shaders/athenea/lod/lod_tree_cut.slang`, `lod_pages.slang`, `occlusion_mask.slang`), y la regla $O<2\sqrt E$ en `apps/athenea/src/CmdMesh2Splat.cpp`. Lo que no ha cambiado es el `CutSelector` del motor (`modules/lod/src/Lod.cpp`, `shaders/athenea/lod/lod_cut.slang`), que sigue siendo el cut por tamaño sobre el octree de §6.1: el render en tiempo real del motor no recorre todavía merge trees. El recorrido de trees en tiempo real vive en el player sparkwebGPU (§7), fuera de este repositorio.

*Referencias:* código: `modules/lod/src/Lod.cpp` (`CutSelector`), `shaders/athenea/lod/lod_cut.slang`, `shaders/athenea/lod/lod_prune.slang`, `shaders/athenea/splat/splat_project.slang` (`kSheetAlpha`); `docs/decisions.md` («Levels of detail» → «Not done yet»).
:::
