<!-- Copyright (c) 2026 jesus luque. -->

# 3. De mallas USD a gaussianas (mesh2splat)

La conversión transforma un stage USD de mallas poligonales y sus materiales en una nube de gaussianas 3D
anisótropas que el resto del sistema dibuja, hornea, diezma y transmite. Se invoca como `athenea mesh2splat STAGE` y
es el núcleo del producto: todo lo que el teléfono acaba reproduciendo fue antes un triángulo de una malla. Esta
sección describe cómo se lee el stage, qué tamaño tiene la celda de una gaussiana, cómo se coloca, se da forma y se
colorea cada gaussiana, cómo trata la conversión los lugares donde una superficie termina o se pliega, cómo captura
un material físico por capas, cómo trata el vidrio y qué se midió sobre el asset del Chevrolet Corvette C7 que guía
la mayor parte del proyecto. El horneado por trazado de caminos que sigue a la emisión (la *transferencia*) es objeto
de la sección 4 y aquí solo se toca donde condiciona la conversión.

Las cifras de esta sección proceden de `docs/decisions.md` y `docs/pieces.md` del repositorio, de los registros de
conversión en `~/luc/athenea-renders/` y del código fuente (`plugins/mesh2splat/mesh2splat.slang`,
`apps/athenea/src/CmdMesh2Splat.cpp`, `shaders/athenea/usd/mesh2splat_*.slang`). Todas las medidas se tomaron en un
Apple M5 Pro (Metal) salvo que se indique otra cosa.

## 3.1 Origen, y dónde se ejecuta la conversión

El algoritmo de emisión es el de *mesh2splat* de Electronic Arts (BSD-3): cada triángulo se proyecta sobre el plano
de ejes hacia el que más apunta su normal, la proyección se rasteriza a una resolución fija y cada téxel cubierto se
convierte en una gaussiana tendida en el plano del triángulo, tan ancha como un téxel medido en el mundo. El
original es un par de shaders de geometría y de fragmentos en OpenGL; aquí es un único kernel de cómputo en el que
cada hilo posee un triángulo y recorre las celdas de su caja proyectada.

El kernel no es un módulo del motor sino un efecto AOFX, el bundle `plugins/mesh2splat`, cargado mediante el mismo
registro de host que comparten todos los programas (la ABI de AOFX, sección 2.14). La conversión es un kernel sin estado
sobre triángulos, que es exactamente la forma de un efecto AOFX, y hacerla uno mantiene un conversor fuera de los
módulos del renderizador y permite que el mismo bundle se ejecute en otros hosts AOFX. El coste es una frontera: a un
efecto se le entregan imágenes, de modo que una malla viaja como una *imagen de números*, seis `float4` por
triángulo — $(\mathbf p, u)$ y $(\mathbf n, v)$ para cada vértice, en espacio de mundo (`mesh_pack.slang`) — y cada
textura que nombra un material viaja como filas de `float4` lineales muestreadas de la tabla de texturas del motor
(`texture_rows.slang`). Ninguna de las dos es una copia: las imágenes viven en el almacenamiento de imágenes de gpe,
que adopta el dispositivo Metal del motor, y `Context::renderView` da una vista de los mismos bytes, así que el kernel
de empaquetado escribe donde el efecto lee. Los registros que escribe el efecto se reúnen en los búferes de
dispositivo de la nube (`mesh2splat_gather`), los límites de la nube se pliegan en el dispositivo
(`mesh2splat_span`) y el horneado los lee en su sitio. Solo los contadores y los arrays USD finales cruzan al
procesador.

El comando es, por tanto, una cadena de pasadas en el dispositivo orquestada por el host:

1. leer el stage (`usd::MeshStage`) y dividir las mallas en *piezas*;
2. empaquetar cada pieza en una imagen de malla, plegar las cajas y elegir la celda de cada pieza (`mesh2splat_cells`);
3. marcar las aristas duras y las cáscaras cerradas por malla (`mesh2splat_edges`) y clasificar el vidrio (`mesh2splat_sheet`);
4. ejecutar el efecto Mesh2Splat por pieza — contar, explorar, emitir — hacia registros;
5. opcionalmente, hornear (radiancia, o una transferencia) en cada gaussiana con el trazador de caminos;
6. escribir un stage `ParticleField3DGaussianSplat` (o un `.athc`) de forma atómica.

## 3.2 Lectura del stage

`usd::MeshStage` recorre el stage directamente con `UsdShade` en lugar de a través de Hydra: la conversión no quiere
ningún índice de render, y el lado de los materiales se apoya en los propios bindings del stage.

**Selección.** `--prim P` restringe la conversión a las mallas en `P` o bajo él; `--hide P` (repetible) escribe
invisibilidad de sesión en `P` y en todo lo que cuelga de él antes de la lectura. Una malla que resulta invisible en
el código de tiempo de la lectura se omite, de modo que la conversión convierte lo que dibuja el renderizador.
`--hide` afecta solo a la conversión: el horneado traza el stage entero, así que una parte convertida con todas las
demás ocultas sigue recibiendo sus sombras e interreflexiones. Esto es lo que permite convertir el Corvette por
grupos (sección 3.11).

**Instanciado.** La malla de un prototipo de `PointInstancer` se emite una vez por instancia del instanciador más
interno que la contiene, en (malla relativa a la raíz de su prototipo) × (transformación de la instancia, raíz del
prototipo incluida) × (instanciador a mundo); los búferes de la malla construida se comparten entre las copias. Antes
de esto, los dieciséis peones del ajedrez se convertían en dos. Los instanciadores anidados solo se expanden en el
nivel más interno.

**Tiempo y pose.** `--time T` fija el código de tiempo en que se leen posiciones, transformaciones y primvars (sus
índices incluidos) y en que traza el horneado. Los `points` de una malla con piel no se animan en USD, así que el
stage se posa primero con `UsdSkelBakeSkinning` en la capa de sesión sobre $[T, T]$, lo que cuesta una pose y deja el
fichero intacto; un stage sin `SkelRoot` queda igual y se convierte idéntico byte a byte. Con `--skinned` las
gaussianas se construyen en la pose de enlace y cada una conserva cuatro articulaciones y los gradientes de sus
pesos; ese camino se describe en la sección 5.

**Piezas.** Una malla cuyos GeomSubsets `materialBind` enlazan materiales distintos se convierte como varias
*piezas*: los triángulos de cada subconjunto y los que ningún subconjunto reclama. Los triángulos de una pieza se
listan en el dispositivo en el orden de la malla (marca, suma prefija, dispersión) y se empaquetan como si fueran una
malla, de modo que el efecto no cambia. Una malla de prueba con dos caras enlazadas en rojo y azul por subconjuntos y
la malla enlazada en verde se convierte en 272 gaussianas rojas y 272 azules y ninguna verde; antes del cambio era
toda verde.

**Los materiales en sus propios términos.** La superficie enlazada — MaterialX `standard_surface`, OpenPBR
`open_pbr_surface`, `gltf_pbr` o `UsdPreviewSurface` — se lee con sus propios nombres de entrada y sus propios valores
por defecto (standard_surface gris 0.8 a rugosidad 0.2, OpenPBR rugosidad 0.3, glTF un metal totalmente rugoso,
UsdPreviewSurface 0.18 a 0.5), siguiendo las conexiones a través de grafos de nodos y entradas de interfaz hasta el
nodo de imagen que produce un valor. Un valor calculado por un grafo (una mezcla, un ruido) no tiene respuesta sin
sombrear un punto, así que se mantiene el valor por defecto y se imprime un aviso; un horneado TX lee después esos
campos desde el material compilado en el dispositivo. El espacio de color de una textura se toma del metadato
`colorSpace` de la entrada de fichero (o de `sourceColorSpace`), y los mapas se limitan a `--texture-size` (1024 por
defecto): un mapa es una imagen `float4` en el dispositivo, así que un mapa de 4k ocuparía 268 MB.

## 3.3 El tamaño de una celda

Todo lo que viene después escala con la celda $h$, la longitud en el mundo que representa una gaussiana. La
conversión ofrece cuatro maneras de elegirla, en orden creciente de preferencia.

**Por modelo** (`--density per-model`, el valor por defecto cuando no se da una cámara). $h = R / N$, donde $N$ es
`--resolution` (512 por defecto) y $R$ el mayor de los dos rangos del plano de proyección de la caja del modelo — los
`orthogonalUvs` del original. De ello se seguían dos defectos: un suelo de 16 unidades diluía el coche apoyado en él,
y un panel orientado hacia delante recorría una celda 2.5 veces más fina que el capó, porque su plano de proyección
es el lado corto del coche.

**Por malla** (`--density per-mesh`). $h = L_m / N$ con $L_m$ el lado más largo de la caja propia de la malla, el
mismo para todos sus triángulos (`cellByLongest`), acotado a $[h_{\min}, h_{\max}]$ dados por
`--cell-min`/`--cell-max`; si se dejan a 0 se derivan de la celda del modelo $h_M$: $h_{\max} = h_M$,
$h_{\min} = h_M/8$, una razón de densidad de 64 entre la parte más fina y la más gruesa. Medido en
`MustangHero.usda` (el coche sobre un suelo de 16 unidades) a `--resolution 512`:

| modo | total | parachoques `Object_199` | capó `Kapoot` | suelo |
|---|---|---|---|---|
| por modelo | 375 199 | 762 (celda 0.0313) | 2 326 | 262 144 |
| por malla, por defecto | 934 338 | 3 082 (celda 0.0039) | 20 279 | 262 144 |
| por malla, `--cell-min` = `--cell-max` = 0.03125 | 375 157 | 764 | 2 339 | 262 144 |

La fila de control reproduce el modo por modelo al 0.01 %; la diferencia restante es solo el efecto de la orientación.

**Desde una cámara** (`--cell-from-camera CAM --camera-pixels P`, a partir de Mesh2GS, arXiv 2606.21898). La celda de
cada pieza es lo que cubre un píxel de la cámara `CAM`, de `P` píxeles de ancho, donde la caja de la malla de la
pieza le queda más cerca:

$$
h = \frac{z\,(A/P)}{f}, \qquad z = \max\bigl(\lVert \operatorname{clamp}(\mathbf e, \mathbf b_{\min}, \mathbf b_{\max}) - \mathbf e \rVert,\; z_{\text{near}}\bigr),
$$

con $A$ la apertura horizontal de la cámara, $f$ su distancia focal, $\mathbf e$ el ojo, y el resultado acotado
solo por `--cell-min`/`--cell-max` del usuario. Mesh2GS toma $z_{\min}$ sobre la caja dentro del frustum y sobre las
muestras temporales de la cámara; la distancia euclídea en `--time` es la misma para una caja a la vista y más fina,
nunca más gruesa, para una al lado del frustum. Con dos tarjetas iguales a tres y siete unidades de una cámara de 64
píxeles, la tarjeta cercana recibió 930 gaussianas y la lejana 169: una razón de 5.5 frente a los $(7/3)^2 = 5.4$
esperados. Este es el modo usado en todos los horneados de producción del Corvette (`--cell-from-camera
/World/Camera --camera-pixels` 640, 1280 o hasta 5120).

**El presupuesto.** `--max-splats` (2 M por defecto) se reparte sobre todo el stage. Cada pieza se cuenta primero
(una ejecución del efecto con sitio para una gaussiana, cuyo recuento y exploración cubren todo lo que escribiría una
ejecución); donde el total $W$ supera el presupuesto $B$, la pieza $i$ que quiere $w_i$ recibe
$s_i = \max(1, B w_i / W)$ y recorre una celda $\sqrt{w_i / s_i}$ veces más gruesa. Una celda $k$ veces más gruesa
cubre la superficie con $k^2$ gaussianas menos, así que las cuotas proporcionales y el engrosamiento uniforme son lo
mismo. Con dos tarjetas que quieren unas 1 300 bajo un presupuesto de 800, las cuotas salieron 356 y 361; gastado en
el orden de las mallas, la segunda tarjeta tenía unas 145.

**Límites por triángulo y por ejecución.** Un triángulo recorre como mucho `--max-cells` celdas de su caja proyectada
($2^{18}$ por defecto); las que sobran se cuentan y se avisan. A 4736 px, `Infinite_Black` del Corvette perdió así
359 628 celdas (agujeros visibles) y se volvió a hornear con `--max-cells 1048576`. La imagen de salida de una
ejecución está limitada a 2 M gaussianas (`kRunCeiling`, 192 MB); una pieza que quiere más se convierte por tramos —
el efecto informa del primer triángulo que el techo cortó y el host vuelve a ejecutar desde ahí con `firstTriangle` —,
lo que da el mismo array que habría escrito una sola ejecución.

## 3.4 Una gaussiana por celda

**La proyección.** Para un triángulo $(\mathbf a, \mathbf b, \mathbf c)$ de normal geométrica $\hat{\mathbf n}_g$,
los dos ejes $(i, j)$ hacia los que menos apunta $\hat{\mathbf n}_g$ definen la proyección; un punto se lleva a
$\mathbf o(\mathbf x) = (x_i - b_{\min,i},\, x_j - b_{\min,j}) / R$, con $R$ el rango en uso (el del modelo, el lado
más largo de la malla o la caja derivada de la cámara). Una celda es $1/N$ de ese cuadrado unidad, es decir, $h = R/N$
en el mundo, acotada como en la sección 3.3. Se emite una gaussiana en cada celda cuyo centro cae dentro del
triángulo proyectado, localizada por sus coordenadas baricéntricas $\boldsymbol\lambda$, en el punto del mundo
$\boldsymbol\mu = \lambda_0 \mathbf a + \lambda_1 \mathbf b + \lambda_2 \mathbf c$.

**Los tamaños.** Con $E = [\mathbf b - \mathbf a,\; \mathbf c - \mathbf a]$ y $O = [\mathbf o_b - \mathbf o_a,\;
\mathbf o_c - \mathbf o_a]$, el jacobiano de la inversa de la proyección es $J = E\,O^{-1} = [\mathbf j_u,
\mathbf j_v]$ (protegido con $|\det O| > 10^{-24}$). Las dos desviaciones típicas en el plano y la tercera son

$$
s_u = \min\!\left(\lVert \mathbf j_u \rVert\,\sigma\,\tfrac{1}{N},\; \ell_{\max}\right), \qquad
s_v = \min\!\left(\lVert \mathbf j_v \rVert\,\sigma\,\tfrac{1}{N},\; \ell_{\max}\right), \qquad
s_n = \varphi\,\min(s_u, s_v),
$$

donde $\sigma$ es `--sigma`, $\ell_{\max}$ la arista más larga del triángulo y $\varphi$ `--flatness`. Tres detalles
se leyeron del código original en lugar de suponerse: el jacobiano es sobre las coordenadas triplanares, no sobre
las coordenadas de textura; el factor $1/N$ lo aplica el exportador del original, no ningún shader (sin él cada
gaussiana tiene el tamaño del modelo); y el marco no son las columnas del jacobiano sino la arista más larga del
triángulo $\hat{\mathbf t}$, la normal de sombreado interpolada $\hat{\mathbf n} = \operatorname{normalize}(\sum_k
\lambda_k \mathbf n_k)$ y su producto vectorial. La covarianza es, por tanto,

$$
\Sigma = R_{\text{frame}}\,\operatorname{diag}(s_u^2, s_v^2, s_n^2)\,R_{\text{frame}}^{\!\top}, \qquad
R_{\text{frame}} = [\hat{\mathbf t}_\perp,\; \hat{\mathbf n}\times\hat{\mathbf t}_\perp,\; \hat{\mathbf n}],
$$

con $\hat{\mathbf t}_\perp$ la arista hecha ortogonal a $\hat{\mathbf n}$. La cota $\ell_{\max}$ se añadió después de
que 1 859 astillas de un BMW, vistas casi de canto por su propia proyección, produjeran columnas del jacobiano veinte
órdenes de magnitud demasiado largas y dibujaran púas blancas de metros de largo.

**Anchura.** El $\sigma = 0.65$ del original deja una superficie convertida transparente en torno a un 30 % para un
trazador de rayos. Medido en un cuadrado unidad iluminado de frente (píxel de la malla 0.5032):

| $\sigma$ | cobertura | píxel |
|---|---|---|
| 0.65 | 0.702 | 0.351 |
| 0.85 | 0.848 | 0.425 |
| 1.0 | 0.911 | 0.456 |
| 1.2 | 0.957 | 0.479 |
| 1.5 | 0.986 | 0.493 |

El valor por defecto es, por tanto, $\sigma = 1.0$; `--sigma 0.65` recupera el original.

**Planitud y surfels.** El original escribía $10^{-7}$ unidades del modelo como tercer tamaño, lo que hacía depender
la delgadez de una gaussiana de la escala del modelo: un peón de 66 mm se trazaba bien mientras que un zorro de 100
unidades se convertía en un fantasma bajo el trazador de rayos. El tercer tamaño es ahora la fracción $\varphi$ (0.1
por defecto) del menor de los tamaños en el plano. `--flatness 0` escribe surfels — gaussianas con una escala
exactamente cero, el disco del splatting gaussiano 2D —, que el rasterizador dibuja mediante una intersección
rayo–plano por píxel y que el trazador de caminos acota a $10^{-4}$ del eje más ancho.

**La opacidad como cobertura.** Un punto de una superficie convertida queda bajo varias gaussianas a la vez, así que
una opacidad escrita directamente en cada una cubre de más: una máscara de 0.5 cubría el 96 % de lo que había detrás.
Toda opacidad que lee la conversión (`--opacity`, la constante del material, el valor de un mapa, lo que conserva un
vidrio) se trata como la *cobertura* $F$ de la superficie y se convierte en el $\alpha$ de cada gaussiana
desarrollando la transmitancia de la pila hasta segundo orden: sobre la retícula de celdas las huellas suman
$2\pi\sigma^2$ y sus cuadrados $\pi\sigma^2$, de modo que

$$
\tfrac{\pi\sigma^2}{2}\,\alpha^2 + 2\pi\sigma^2\,\alpha = -\ln(1 - F)
\;\;\Longrightarrow\;\;
\alpha = \frac{\sqrt{4S^2 + 2S\,L} - 2S}{S} + \tfrac{1}{255}, \quad S = \pi\sigma^2,\; L = -\ln(1-F),
$$

donde el $1/255$ devuelve el área que pierde una gaussiana al cortarse en $1/255$; $F \ge 0.999$ se lleva a 1 y 0 a
0. Medido en planos de cobertura constante 0.25 / 0.5 / 0.75, el rasterizador da 0.251 / 0.501 / 0.752 y el trazador
de rayos lo mismo a tamaño completo, frente a 0.253 / 0.510 / 0.770 con solo el término de primer orden. Un mapa de
recorte por debajo de `--opacity-cut` (0.5) no escribe ninguna gaussiana, así que el presupuesto va donde está la
superficie.

**Un triángulo que no atrapa ninguna celda.** Un triángulo más pequeño que una celda puede no contener ningún centro
de celda; rasterizado, desaparece. Convertido, eso eliminaba la rejilla y las gomas de las ventanas del Buick (80 137
triángulos se convertían en 34 011 gaussianas). Ahora tal triángulo recibe una gaussiana en su baricentro,
dimensionada por sus propios segundos momentos (`m2sOwnSize`): la covarianza del triángulo uniforme
$\Sigma_\triangle = \tfrac{1}{12}\sum_k \mathbf d_k \mathbf d_k^\top$ (vértices respecto al baricentro, en el plano
tangente), ensanchada como se ensancha una celda, $\Sigma = 12\sigma^2 \Sigma_\triangle$, con el eje ancho acotado a
$[0.05\,s_{\max}, s_{\max}]$ del tamaño de la celda y el estrecho a al menos 0.02 del ancho. Antes de esto, el labio
enrollado de un panel — una tira de triángulos menores que la celda — dibujaba una banda de cuentas del ancho de una
celda, de tres píxeles de grosor, a lo largo de cada arista del Corvette donde el trazador de caminos dibuja una
línea finísima. En los horneados del Corvette, el vidrio y los neumáticos salen a aproximadamente una gaussiana por
triángulo tanto a 640 como a 1280 px.

**Orden determinista.** El original añadía con un contador atómico, así que dos conversiones del peón producían las
mismas 729 073 gaussianas en órdenes distintos. La emisión son ahora tres pasadas — contar las celdas cubiertas por
cada triángulo, una exploración exclusiva en un grupo de trabajo de 256 hilos, escribir las gaussianas de cada
triángulo en su desplazamiento — que comparten una función (`m2sTriangleOf`), de modo que contar y escribir no
pueden discrepar. Dos conversiones son ahora idénticas byte a byte, un presupuesto conserva las primeras gaussianas
en el orden de la malla y la gaussiana $i$ es la misma gaussiana de una pose a la siguiente. Un registro que se
decodifica en nada se escribe como una ranura vacía en lugar de descartarse, de modo que los índices de los arrays
siguen alineados.

## 3.5 Donde una superficie termina o se pliega

Una gaussiana mide una celda de ancho, y en el Corvette una celda es un píxel de la cámara de conversión. Una parte
más estrecha que eso — la pestaña que un capó dobla hacia abajo dentro de su junta — seguía recibiendo gaussianas
del ancho de una celda en su propio plano; asomaban fuera de la junta sobre la pintura de al lado, llevando la
oscuridad horneada de la junta, y cada borde de panel se convertía en una línea negra gruesa y ondulada donde el
trazador de caminos dibuja una fina.

**Aristas duras** (`mesh2splat_edges.slang`). Por malla, sobre todas sus piezas, cada arista y cada vértice se
indexan por la posición de sus puntos y se ordenan por radix en el dispositivo; cada tramo de claves de arista
iguales es una arista. Una arista es *dura* cuando está abierta (la usa un solo triángulo: la parte termina o empieza
otra malla), cuando los dos triángulos que la comparten se pliegan más de $40^\circ$ (`kCreaseDegrees`) o cuando la
comparten más de dos triángulos (no variedad). Como las claves son posiciones sobre toda la malla, la frontera entre
dos materiales sobre una misma superficie suave no es una arista. Cada arista dura se convierte en una recta con un
interior, $(\hat{\mathbf m}, c)$ con interior $\hat{\mathbf m}\cdot\mathbf x \ge c$ — el plano que pasa por la
arista, ortogonal a su triángulo, orientado hacia el tercer vértice —, que su propio triángulo conoce como un bit y
que se comunica a cada triángulo que toca cualquiera de sus extremos (hasta cinco rectas cada uno), puesto que una
gaussiana cerca de un vértice puede quedar junto a la arista dura del triángulo vecino.

**Contener el alcance** (`m2sHoldEdges`, `--edge-reach` $\rho$, 1.5 por defecto). Para cada recta contenida en el
plano de la gaussiana, sea $\mathbf m$ su normal en la base tangente, $C$ la covarianza en el plano,
$r^2 = \mathbf m^\top C\,\mathbf m$ la varianza a través de la recta y $d$ la distancia con signo del centro a ella.
El tamaño permitido a través de la recta es $a = \max(d/\rho,\; 0.05\,r)$, y donde $r > a$

$$
C \leftarrow C - \kappa\,(C\mathbf m)(C\mathbf m)^\top, \qquad \kappa = \frac{1 - a^2/r^2}{r^2},
$$

lo que hace $\mathbf m^\top C\,\mathbf m = a^2$ exactamente y conserva la varianza a lo largo de la dirección
conjugada de la recta, de modo que una tira de gaussianas sigue cubriendo la tira. Se ignoran las rectas que no están
en el plano de la gaussiana (la pestaña vista desde la pintura de la que se dobla) y aquellas tras las que el centro
queda a más de un cuarto de su tamaño; las rectas se aplican en un orden canónico, de modo que el resultado no
depende del orden en que se encontraron. La nueva covarianza se rediagonaliza en una tangente y dos tamaños. El
número de gaussianas no cambia; solo su forma.

**Cáscaras cerradas.** La misma pasada cuenta las aristas no compartidas exactamente por dos triángulos, un contador
leído por malla. Una malla se marca como cerrada cuando

$$
n_{\text{open}} < 2\sqrt{1.5\,T},
$$

con $T$ sus triángulos ($1.5T$ aproxima sus aristas soldadas), y sus gaussianas llevan
`primvars:athenea:splat:closed = 1`, lo que permite a un reproductor o a un corte descartar nodos que miran hacia
atrás. Primero se probó una proporción fija de aristas (2 %) y falló: el borde de una lámina abierta crece como la
raíz cuadrada de su área, así que un tubo lo bastante fino pasaba por cerrado — los cilindros de las llantas (1 %) se
descartaban y se veía luz entre los radios en el teléfono. Frente a la raíz cuadrada, las puertas (ninguna arista
abierta), los logotipos (1.4) y las pinzas (1.5) siguen cerrados y las llantas (4.0), los neumáticos (3.1), el capó
(3.1) y el parachoques (4.4) no: 102 de las 192 mallas del Corvette. La puerta de un coche son dos pieles; dibujada
sin descarte, la piel interior se ordenaba delante de la exterior a parches, una red de grietas oscuras al tamaño del
teléfono.

**Medido.** La base TX del Corvette a 640 px de `/World/Camera`, horneada de nuevo con las aristas contenidas (1.5 por
defecto) y, como control, con `--edge-reach 0`, ambas de 5 380 804 gaussianas; fotograma diurno a 1920 × 1080 frente
al path trace de la malla:

| horneado | relMSE (HDR) | PSNR (pantalla) | SSIM |
|---|---|---|---|
| base anterior (build antigua) | 0.0900 | 21.83 dB | 0.8555 |
| control, aristas sin contener | 0.0902 | 21.86 dB | 0.8554 |
| aristas contenidas, $\rho = 1.5$ | 0.0887 | 21.93 dB | 0.8567 |

La ganancia global es pequeña porque las aristas son una fracción pequeña del fotograma; el efecto visible está en
las juntas de los paneles.

<!-- /Users/muriel/luc/athenea-renders/simplify3/edges/edges_cmp.png -->
<!-- /Users/muriel/luc/athenea-renders/simplify3/edges/day_new_Camera.png -->
<!-- /Users/muriel/luc/athenea-renders/simplify3/edges/day_ctl_Camera.png -->

## 3.6 Captura del material

Cada registro de gaussiana lleva la posición, la opacidad derivada de la cobertura, tres tamaños, una rotación y un
color, seguidos — según lo que necesite la nube — de la normal de sombreado y el metálico, la rugosidad, la
transmisión y las coordenadas de textura; cuatro articulaciones y sus gradientes; tres entradas de relieve para el
desplazamiento; una entrada de emisión; cuatro entradas de capas; y el operador de forma.

**Base.** El color es el color base del material por su peso base, por el mapa de albedo donde lo haya (decodificado
a lineal). El metálico y la rugosidad son las constantes del material, multiplicadas por los canales del mapa
empaquetado donde lo haya (azul metálico, verde rugosidad), como glTF multiplica sus factores; un canal ausente es un
factor uno. El original tomaba por defecto $(0.1, 0.5)$ solo del mapa, así que todo material sin mapa se convertía
en el mismo plástico; pasando los valores propios del material, un cuadrado convertido bajo una luz de frente da
0.5009 frente a 0.5032 de la malla.

**Normal de sombreado.** Un mapa de normales se lee en el marco que dan sus UV — $\partial\mathbf p/\partial u$ a
partir de las posiciones y las UV del triángulo, ortonormalizado frente a la normal, como hace el shader de la malla
— y el resultado se guarda como normal de sombreado de la gaussiana (`primvars:athenea:splat:normal`) junto a su
marco geométrico. Antes el mapa se leía respecto a la arista más larga, y cada faceta del mármol del peón giraba sus
vetas según su propio ángulo. El disco en sí no se gira salvo que se pase `--normal-map-turns`, lo que abriría la
superficie donde el relieve es pronunciado.

**Capas** (propuesta 026). Una gaussiana lleva el peso, el color y el índice especulares; el peso, la rugosidad y el
índice de la laca; el color del brillo aterciopelado (sheen) por su peso y su rugosidad; y el interruptor de
oscurecimiento de la laca — en las unidades de OpenPBR, tres palabras por splat en el dispositivo. Hasta tres mapas
sobre estas entradas se muestrean por gaussiana (`m2sLayersAt`). Hacía falta porque la pintura del Corvette
(`Car_Paint_Main`) es OpenPBR de metalicidad 1 sobre una base de aproximadamente 0.05 bajo una laca transparente de
peso 1: todo lo que el ojo ve de ella es la laca, y sin la laca un coche convertido salía casi negro.

**Metales de Schlick.** Los metales de OpenPBR y glTF son el `generalized_schlick_bsdf` de MaterialX (el color de
frente, el color especular a $82^\circ$), no un conductor del índice artístico; para un metal oscuro ambos difieren
en un factor dos a ángulos rasantes (a $60^\circ$: 0.172 el conductor, 0.077 el Schlick desde $F_0 = 0.047$). El tipo
de material se escribe por gaussiana (`primvars:athenea:splat:schlickMetal`) y el renderizador lo refleja con el
ajuste DFG de sus dos extremos. Una variante anterior de la misma confusión hacía que el horneado descartara todo
metal OpenPBR como pulido dieléctrico: el parachoques delantero de un Mustang guardaba en sus 27 016 gaussianas la
constante que se decodifica en negro; conservando como cuerpo los lóbulos de Schlick con $F_0 > 0.2$, el color medio
del parachoques pasó a ser (0.45, 0.51, 0.61).

**Emisión y transmisión.** La emisión es el color del material por su mapa, lineal y sin acotar, en una entrada
propia. La transmisión $T$ y el color de transmisión $\mathbf c_T$ son un canal que el original no tenía (el vidrio
salía blanco opaco). El color se lleva hacia el color de transmisión,

$$
\mathbf c \leftarrow \operatorname{lerp}\!\bigl(\mathbf c,\; \operatorname{lerp}(\mathbf 1, \mathbf c, w_T)\,\mathbf c_T,\; T\bigr),
$$

con $w_T = 1$ para glTF (transmisión teñida por el color base) y 0 para OpenPBR y standard_surface. La opacidad es la
correspondencia de cobertura de la sección 3.4 aplicada a $o\cdot o_{\text{mat}}\cdot\text{máscara}\cdot
\operatorname{lerp}(1, g, T)$, siendo $g$ lo que aún cubre una superficie totalmente transmisora (sección 3.8).

**La marca de rugosidad del material.** La rugosidad guardada puede ensancharse por el filtro especular (sección
3.7). La rugosidad propia del material antes del ensanchamiento se escribe aparte,
`primvars:athenea:splat:materialRoughness` (−1 bajo un mapa de rugosidad, que se lee como nítida). El reproductor la
usa para decidir si el lóbulo de pulido responde su parte ocluida desde el campo reflejado horneado (rugosidad del
material de al menos 0.3: pintura, 0.34) o la deja oscura (cromo 0, metal rugoso 0.22); en el árbol diezmado se
empaqueta en la palabra `pbr` (bit 26 presente, bits 27–31 el valor en cinco bits).

## 3.7 El filtro especular (ensanchamiento de Toksvig)

Una gaussiana tiene una sola normal. Donde una tira de cromo se curva bajo ella, el píxel trazado promedia un
reflejo que se desplaza por la huella y muestra una línea fina; la gaussiana muestra la luz que encuentra su única
normal, a pleno brillo, sobre toda su huella. En el Corvette, el embellecedor de la rejilla — de cuatro píxeles de
alto — volvía como una fila de cuentas blancas, y el cromo daba 1.6 veces el path trace mientras que una
bola de cromo se validaba dentro del 1 %.

La conversión ensancha por ello la rugosidad según la dispersión de normales bajo la gaussiana (`m2sSpread`,
`m2sWidened`). La tasa de giro es el mayor ángulo entre normales de vértice por unidad de distancia entre vértices,
sobre las tres aristas del triángulo, y la dispersión es esa tasa por el mayor tamaño de la gaussiana:

$$
s = k\,\max_{e}\frac{\angle(\mathbf n_{e_0}, \mathbf n_{e_1})}{\lVert \mathbf p_{e_1} - \mathbf p_{e_0} \rVert}\,\max(s_u, s_v),
$$

con $k$ = `--specular-filter` (1 con `--transfer`, 0 en otro caso). Con la rugosidad perceptual $r$ y el
$\alpha = r^2$ de GGX, la dispersión se suma a la varianza de las pendientes (Toksvig):

$$
\alpha'^2 = \alpha^2 + 2 s^2, \qquad r' = \operatorname{saturate}\!\left((r^4 + 2 s^2)^{1/4}\right).
$$

Un triángulo plano, o uno cuyos vértices comparten normal, queda igual. La rugosidad de la laca se ensancha del mismo
modo salvo que la nube lleve el operador de forma.

**El operador de forma.** Para una transferencia, la conversión escribe también, en los ejes propios de la gaussiana
$(\mathbf u, \mathbf v)$, el operador de forma simétrico $S$ con $\mathrm d\mathbf n = S\,\mathrm d\mathbf x$,
resuelto a partir de las diferencias de normales de los vértices frente a sus diferencias de posición (`m2sShape`,
guardado como $(S_{uu}, S_{uv}, S_{vv})$). El renderizador lo usa para girar el reflejo a través de la huella píxel a
píxel (el «lóbulo nítido» por píxel). Ensanchar también la laca difuminaría dos veces, así que donde se escribe $S$
la laca no se ensancha, y el reproductor lee los lóbulos nítidos con la rugosidad sin ensanchar
$r_0 = \max(r'^4 - 2s^2, 0)^{1/4}$ (`unwidened`), con $s$ guardada por gaussiana.

**Medido.** A 640 px la laca salía con rugosidad 0.17–0.24 y un 5–14 % de paso alto en los pliegues; a 1280 px la
dispersión bajo cada gaussiana es menor y la rugosidad de la laca sale exactamente 0, así que el ensanchamiento
importa menos cuanto más fino es el horneado. Aun así, quitarlo es peor (sección 3.11): con `--specular-filter 0`
aparece una línea oscura discontinua en el pliegue del faldón y la tira de cromo centellea.

## 3.8 Vidrio

Una gaussiana no puede refractar, pero puede dejar pasar lo que tiene detrás mediante su opacidad, reflejar con su
lóbulo especular y llevar un índice de refracción según el cual el renderizador desvía
(`primvars:athenea:splat:ior`, uno por nube; con capas, cada gaussiana desvía según su propio índice especular). La
tarea de la conversión es decidir, para cada pieza transmisora, si el vidrio es un *sólido* (un cuerpo: una bola, una
lente) o una *pared fina* (una lámina o placa cuyas dos caras no desvían nada), porque ambos se dibujan de forma
distinta.

**Clasificación** (`classifySheets`, `mesh2splat_sheet.slang`), para cada pieza transmisora cuyo material no se
declara ya de pared fina. En el dispositivo, las aristas de la pieza se indexan por posición, se ordenan y se cuentan
sus tramos; se suman el volumen con signo $V$ (tetraedros al origen) y el área $A$. Entonces

- *abierta*: más de $1/200$ de las aristas tienen un solo triángulo, o alguna arista tiene más de dos;
- *placa*: cerrada, y el grosor $2V/A$ está por debajo de cuatro celdas del modelo o de $0.02\sqrt A$
  ($2V/A$ es la separación de una placa y dos tercios del radio de una bola).

Una pieza abierta o placa se lee como pared fina — salvo que esté *tintada* (luminancia del color de transmisión por
debajo de 0.9), que de momento sigue siendo sólida, ya que una pared fina deja pasar el fondo en gris. El vidrio del
Corvette resultó ser placas cerradas de 3–4 mm de grosor (propuesta de investigación 055), que la regla de aristas
por sí sola llamaba sólidas; la regla del grosor las reconoce. Cada decisión se imprime, p. ej. para la lente del
piloto trasero: «tinted, kept solid glass: 200 of its 17284 edges open, 0 shared by more than two; 1.199 thick (2V/A)
against a cell of 0.007921». `--thin-glass MAT` y `--solid-glass MAT` la anulan por material.

**Cobertura.** Un vidrio sólido cubre `--glass-opacity` de lo que tiene detrás (0.6 por defecto; una lente TX responde
su fondo solo a partir de las cúpulas, así que un vidrio que cubriera por completo ocultaría cualquier otra luz que lo
atraviese). Una pared fina cubre exactamente lo que la lámina refleja de frente, ambas caras con interreflexión:

$$
g_{\text{thin}} = \min\!\left(\frac{2R}{1+R},\; 0.998\right), \qquad R = \left(\frac{n-1}{n+1}\right)^2,
$$

y el fotograma escala su reflejo de vuelta por $1/R$. Con el índice del parabrisas, 1.16, eso es una opacidad de
0.0055, y el rasterizador dibuja estas láminas con una opacidad de al menos 0.1 y el color reducido en el mismo
factor, lo que mantiene intacto el producto $\alpha\,\mathbf c$ — el reflejo — (relMSE del parabrisas 2.26 → 0.89 bajo
el cielo del taller).

## 3.9 Captadores de sombra y de luz

El suelo no se convierte como superficie; lo que se convierte es su interacción con el objeto, de modo que la
reproducción es solo de gaussianas.

**Captador de sombra** (`--shadow-catcher --prim OBJETO`). El suelo se encuentra automáticamente (la mayor malla plana
fuera del objeto cuya parte superior está en la base del objeto; `--catcher-ground` nombra una). Un parche de
cuadriláteros — la huella del objeto ensanchada `--catcher-margin` alturas (1.5), dividida en una retícula para que
ningún triángulo supere `--max-cells` — se añade a una copia de sesión del stage y se convierte solo, con una celda de
una centésima de la altura del objeto, como una transferencia TX cuyos rayos encuentran el objeto y el suelo. Al
dibujar, la opacidad de la gaussiana es $-\ln(\text{razón})/2\pi$, donde la razón es la luz que le llega con el objeto
frente a sin él, ya que una retícula de gaussianas de $\sigma = 1$ celda suma $2\pi$ y una capa de opacidad $a$ deja
pasar $e^{-2\pi a}$. El del Corvette: un parche de 5.9 × 8.3 m, 11 × 11 cuadriláteros, 309 k gaussianas con una celda
de 1.26 cm, horneado en 79 s. Frente al path trace:

| | bajo el coche (GT 0.182) | contacto del parachoques (GT 0.043) | suelo abierto (GT 0.217) |
|---|---|---|---|
| mapa de sombras propio del suelo de malla | 0.182 | 0.041 | 0.215 |
| captador, profundidad 2.4, dos triángulos | 0.144 | 0.002 | 0.206 |
| captador, profundidad $2\pi$, retícula | 0.192 | 0.030 | 0.220 |

**Captador de luz** (`--light-catcher`). El mismo parche, alcanzando `--light-reach` longitudes del objeto (3) por
delante y por detrás, con una celda de una treintava parte de la altura, convertido como un horneado de radiancia por
grupo de luces con la cúpula en negro; dibujado de forma aditiva, lleva la mancha de la luz de cruce y el resplandor de
los pilotos traseros.

## 3.10 Validación frente al path trace

`--validate DIR` mide una conversión material a material frente al stage trazado por caminos, para cualquier stage.
Los grupos son los bindings de material del stage (una malla de dos materiales está en ambos); la conversión es la
del propio comando, ejecutada una vez por grupo con todas las mallas ajenas a él ocultas, de modo que cada opción es
lo que se mide. La referencia se traza una vez (`--validate-paths` 512, `--validate-bounces` 6) y se conserva; el
stage se rasteriza una vez como mallas con su Cryptomatte, y la máscara de cada grupo es la cobertura de cada rango
cuyo id es el hash de una de sus mallas. Dentro de la caja de la máscara, el efecto Measure da relMSE, p99 y las
medias de la nube y del rasterizado de la malla a su lado, escritos en `validate.json`. `--validate-sky white` (o una
imagen) dibuja todo bajo otro cielo con las demás luces del stage apagadas, de modo que el error de una transferencia
es solo el de su oclusión y su reflexión. Es lo que reveló, entre otras cosas, que una bola de vidrio sólido se
trazaba a la mitad del brillo del path trace (0.096 frente a 0.195) antes de que los rayos se desviaran hacia
su interior, y que la bola de pintura bajo una cúpula blanca daba un 13 % por debajo a causa del Fresnel de la laca.

## 3.11 Resultados medidos sobre el Corvette

Todos los horneados siguientes son transferencias TX, `--cell-from-camera /World/Camera`, 64 caminos por gaussiana,
3 rebotes, sin caminos adaptativos extra, con el filtro especular y el alcance de aristas en sus valores por defecto
(1 y 1.5). «Recorte del coche» es el PSNR de pantalla sobre el recorte de 900 × 420 del coche, y «fotograma» sobre el
fotograma completo de 1920 × 1080, ambos frente al path trace de la malla, tras cortar el árbol de fusión a
0.25 px con descarte de caras traseras.

**640 → 1280 px.** Las hojas del horneado de 640 px medían 3–4 mm en el capó, y las rayas de la puerta y los pelos
del faldón — ausentes en el path trace — estaban en la nube. El coche se volvió a hornear a 1280 px en 8 grupos
de partes de primer nivel equilibrados por recuento, cada uno convertido con todas las demás partes ocultas (nombres
de parte tomados del manifiesto de Cryptomatte). Los ocho horneados escribieron 891 387, 896 496, 809 073, 790 403,
1 631 765, 1 073 954, 1 302 878 y 1 307 473 gaussianas (8 703 429 en total) en 212–304 s de transferencia cada uno,
unos 34 min en conjunto.

| | 640 px | 1280 px |
|---|---|---|
| hojas | 5.38 M | 8.70 M |
| capó / puerta | 50 k / 79 k | 151 k / 266 k |
| vidrio, neumáticos | aprox. una por triángulo | aprox. una por triángulo |
| recorte del coche / fotograma (PSNR) | 16.55 / 22.18 dB | 17.88 / 23.90 dB |

Análisis de la nube de 1280 px frente a la de 640 px: los reflejos del capó pasan de 3.2 a 2.0 px de su posición en el
path trace, el contraste de los reflejos abiertos de la puerta de 0.35/0.53 a 0.56/0.74 del path trace, los
agujeros de la rejilla en el corte grueso de 2.24 a 1.04, y los pelos del faldón de 987 a 459 px solo por la
resolución.

**La prueba A/B/C.** Antes de fijar los ajustes de 1280 px se hornearon ocho partes (ambas puertas, ambos paneles del
faldón, la aleta delantera, la rejilla cromada, el marco de la rejilla y el capó) con todo el coche proyectando
sombra, 931 017 gaussianas en cada variante:

| variante | ajustes | resultado |
|---|---|---|
| **A** | como el horneado de 640 px ($k = 1$, $\rho = 1.5$) | elegida: sin pelos, la puerta lisa |
| B | `--specular-filter 0` | línea oscura discontinua en el pliegue del faldón; la tira de cromo centellea |
| C | B + `--edge-reach 1.0` | alargamiento y pelos idénticos a B |

La hipótesis a prueba era que las rayas de la puerta eran el ensanchamiento de Toksvig de la laca saturándose en los
pliegues y que los pelos del faldón eran las gaussianas alargadas de la pasada de aristas. Ninguna se sostuvo a
1280 px: el ensanchamiento es necesario en los pliegues y un alcance de aristas más estricto no cambió nada, así que se
mantuvo A.

<!-- /Users/muriel/luc/athenea-renders/pieces/car1280/test/A.png -->
<!-- /Users/muriel/luc/athenea-renders/pieces/car1280/test/B.png -->
<!-- /Users/muriel/luc/athenea-renders/pieces/car1280/test/C.png -->

**El nivel de primer plano a 5120 px.** De cerca en el teléfono, la cámara está mucho más próxima que `/World/Camera`,
unos 0.5 mm por píxel del teléfono frente a hojas de 3–4 mm. Una prueba del logotipo y la rejilla desde una cámara
cercana pasó de 108 527 a 404 594 gaussianas entre 1280 y 5120 px y mostró nítidos la bandera a cuadros, la pajarita y
la malla de la rejilla. Las 66 partes a las que el horneado de 1280 px ya había dado como mucho 1.5 gaussianas por
triángulo (insignias, rejillas, faros, pinzas, retrovisores, ruedas, limpiaparabrisas, molduras) se volvieron a
hornear a 5120 px en siete grupos; los paneles, a 5120 px o tan finos como mantuviera cada parte bajo un tope de hojas
(5 M, rebajado a 4.5 M después de que un árbol de 5.01 M no cupiera en la memoria del dispositivo), con
`--max-cells 1048576`:

| grupos | hojas | resolución (px) |
|---|---|---|
| fine0–6 (66 partes detalladas) | 18 610 627 | 5120 |
| puerta (piloto) | 4 084 170 | 5120 |
| Bottom | 4 897 428 | 3392 |
| Part_3 | 4 370 459 | 4480 |
| Infinite_Black | 4 481 738 | 4736 |
| seis paneles grandes y el parabrisas | 15 994 566 | 5120 |
| pan0–4 (paneles pequeños) | 17 088 152 | 5120 |
| Front_Bumper_001 | 5 008 986 | 4480 |
| **24 árboles** | **74 536 126** | |

La transferencia de cada panel grande tardó 2–25 min (p. ej. el capó, 2 264 198 gaussianas en 120 s;
Front_Bumper_001, 5 009 059 en 1 487 s) en una máquina compartida con otros trabajos de GPU, así que no son medidas de
rendimiento. Se escribieron los 24 árboles de fusión (74 328 031 fusiones, 208 095 raíces, 148 864 157 nodos en
49 634 páginas, 87.4 GB), que comparten un codebook de oclusión de 8 182 entradas; la sección 6.9 da los detalles. (La
cuenta horneada de Front_Bumper_001, 5 009 059, es la del registro de la conversión; el árbol tiene 5 008 986 hojas,
la cifra que usa la tabla.)

## 3.12 Limitaciones

- Una gaussiana tiene una sola normal y la huella de una celda: un tubo de cromo de pocos píxeles de ancho sigue
  siendo un 38 % más brillante que el path trace en el parachoques delantero, y el labio inferior del
  parachoques sale irregular.
- Los faros no llevan emisión en el horneado de transferencia; los faros delanteros salen oscuros frente al trazado
  de caminos.
- Un vidrio fino tintado se convierte como sólido, y una nube conserva un único índice salvo que lleve capas.
- La correspondencia de cobertura supone el solapamiento $2\pi\sigma^2$ de la retícula de celdas; los triángulos
  menores que una celda y las celdas desplazadas y divididas se solapan de otra forma y no se corrigen.
- Los instanciadores de puntos anidados solo se expanden en el nivel más interno; el movimiento escrito como
  velocidades se lee en la muestra.
- La memoria del dispositivo limita una parte a unos 5 M de hojas para el árbol de fusión que sigue; las partes
  mayores se convierten a una resolución de cámara más gruesa.
