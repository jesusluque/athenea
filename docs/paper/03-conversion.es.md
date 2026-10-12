<!-- Copyright (c) 2026 jesus luque. -->

# 3. De meshes USD a Gaussians (mesh2splat)

La conversión es la primera etapa del baker de relight: transforma un stage USD de meshes poligonales y sus
materiales en una nube de Gaussians 3D anisótropas, y esa nube es la que el bake de transfer de la sección 4 llena
de luz. Se invoca como `athenea mesh2splat STAGE`; cada Gaussian que el resto del sistema dibuja, hace bake, diezma
y transmite por streaming fue antes un triángulo de un mesh. Esta sección describe cómo se lee el stage, qué tamaño tiene la cell de una Gaussian, cómo se
coloca, se da forma y se colorea cada Gaussian, cómo trata la conversión los lugares donde una superficie termina o
forma un fold, cómo captura un material físico por capas, cómo trata el vidrio y qué se midió sobre el asset del
Chevrolet Corvette C7 que guía la mayor parte del proyecto. El bake por path tracing que sigue a la emisión (el
*transfer*) es objeto de la sección 4 y aquí solo se toca donde condiciona la conversión.

Las cifras de esta sección proceden de `docs/decisions.md` y `docs/pieces.md` del repositorio y del código fuente
(`plugins/mesh2splat/mesh2splat.slang`, `apps/athenea/src/CmdMesh2Splat.cpp`,
`shaders/athenea/usd/mesh2splat_*.slang`). Todas las medidas se tomaron en un Apple M5 Pro (Metal) salvo que se
indique otra cosa.

## 3.1 Origen, y dónde se ejecuta la conversión

El algoritmo de emisión es el de *mesh2splat* de Electronic Arts (BSD-3): cada triángulo se proyecta sobre el plano
de ejes hacia el que más apunta su normal, la proyección se rasteriza a una resolución fija y cada téxel cubierto se
convierte en una Gaussian tendida en el plano del triángulo, tan ancha como un téxel medido en el mundo. El
original es un par de shaders de geometría y de fragmentos en OpenGL; aquí es un único kernel de cómputo en el que
cada hilo posee un triángulo y recorre las cells de su caja proyectada.

El kernel no es un módulo del motor sino un efecto AOFX, el bundle `plugins/mesh2splat`, cargado mediante el mismo
registro de host que comparten todos los programas (la ABI de AOFX, sección 2.14). La conversión es un kernel sin estado
sobre triángulos, que es exactamente la forma de un efecto AOFX, y hacerla uno mantiene un conversor fuera de los
módulos del renderer y permite que el mismo bundle se ejecute en otros hosts AOFX. El coste es una frontera: a un
efecto se le entregan imágenes, de modo que un mesh viaja como una *imagen de números*, seis `float4` por
triángulo — $(\mathbf p, u)$ y $(\mathbf n, v)$ para cada vértice, en espacio de mundo (`mesh_pack.slang`) — y cada
textura que nombra un material viaja como filas de `float4` lineales muestreadas de la tabla de texturas del motor
(`texture_rows.slang`). Ninguna de las dos es una copia: las imágenes viven en el almacenamiento de imágenes de gpe,
que adopta el dispositivo Metal del motor, y `Context::renderView` da una vista de los mismos bytes, así que el kernel
de empaquetado escribe donde el efecto lee. Los registros que escribe el efecto se reúnen en los buffers de
dispositivo de la nube (`mesh2splat_gather`), los límites de la nube se pliegan en el dispositivo
(`mesh2splat_span`) y el bake los lee en su sitio. Solo los contadores y los arrays USD finales cruzan al
procesador.

El comando es, por tanto, una cadena de pasadas en el dispositivo orquestada por el host (figura 5):

1. leer el stage (`usd::MeshStage`) y dividir los meshes en *piezas*;
2. empaquetar cada pieza en una imagen de mesh, plegar las cajas y elegir la cell de cada pieza (`mesh2splat_cells`);
3. marcar las aristas duras y las cáscaras cerradas por mesh (`mesh2splat_edges`) y clasificar el vidrio (`mesh2splat_sheet`);
4. ejecutar el efecto Mesh2Splat por pieza — contar, explorar, emitir — hacia registros;
5. opcionalmente, hacer bake (radiancia, o un transfer) en cada Gaussian con el path tracer;
6. escribir un stage `ParticleField3DGaussianSplat` (o un `.athc`) de forma atómica.

![Figura 5. La conversión como secuencia de pasadas en el dispositivo. Los pasos 2–7 son kernels o el efecto AOFX Mesh2Splat; el paso 8 es el bake del transfer de la sección 4.](figures/fig-mesh2splat.es.svg)

## 3.2 Lectura del stage

`usd::MeshStage` recorre el stage directamente con `UsdShade` en lugar de a través de Hydra: la conversión no quiere
ningún índice de render, y el lado de los materiales se apoya en los propios bindings del stage.

**Selección.** `--prim P` restringe la conversión a los meshes en `P` o bajo él; `--hide P` (repetible) escribe
invisibilidad en `P` y en todo lo que cuelga de él, en una capa no persistente sobre el stage, antes de la lectura. Un
mesh que resulta invisible en el código de tiempo de la lectura se omite, de modo que la conversión convierte lo que
dibuja el renderer. `--hide` afecta solo a la conversión: el bake traza el stage entero, así que una parte convertida
con todas las demás ocultas sigue recibiendo sus sombras e interreflexiones. Esto es lo que permite convertir el
Corvette por grupos (sección 3.11).

**Instanciado.** El mesh de un prototipo de `PointInstancer` se emite una vez por instancia del instanciador más
interno que lo contiene, en (mesh relativo a la raíz de su prototipo) × (transformación de la instancia, raíz del
prototipo incluida) × (instanciador a mundo); los buffers del mesh construido se comparten entre las copias. Sin esta
expansión, los dieciséis peones del ajedrez se convierten en dos. Los instanciadores anidados solo se expanden en el
nivel más interno.

**Tiempo y pose.** `--time T` fija el código de tiempo en que se leen posiciones, transformaciones y primvars (sus
índices incluidos) y en que traza el bake. Los `points` de un mesh con skinning no se animan en USD, así que el
stage se posa primero con `UsdSkelBakeSkinning` en una capa no persistente sobre $[T, T]$, lo que cuesta una pose y
deja el fichero intacto; un stage sin `SkelRoot` queda igual y se convierte idéntico byte a byte. Con `--skinned` las
Gaussians se construyen en la bind pose y cada una conserva cuatro joints y los gradientes de sus pesos; ese camino
se describe en la sección 5.

**Piezas.** Un mesh cuyos GeomSubsets `materialBind` enlazan materiales distintos se convierte como varias
*piezas*: los triángulos de cada subconjunto y los que ningún subconjunto reclama. Los triángulos de una pieza se
listan en el dispositivo en el orden del mesh (marca, suma prefija, dispersión) y se empaquetan como si fueran un
mesh, de modo que el efecto no cambia. Un mesh de prueba con dos caras enlazadas en rojo y azul por subconjuntos y
el mesh enlazado en verde se convierte en 272 Gaussians rojas y 272 azules y ninguna verde; leyendo solo el binding
propio del mesh sería todo verde.

**Los materiales en sus propios términos.** La superficie enlazada — MaterialX `standard_surface`, OpenPBR
`open_pbr_surface`, `gltf_pbr` o `UsdPreviewSurface` — se lee con sus propios nombres de entrada y sus propios valores
por defecto (standard_surface gris 0.8 a roughness 0.2, OpenPBR roughness 0.3, glTF un metal totalmente rugoso,
UsdPreviewSurface 0.18 a 0.5), siguiendo las conexiones a través de grafos de nodos y entradas de interfaz hasta el
nodo de imagen que produce un valor. Un valor calculado por un grafo (una mezcla, un ruido) no tiene respuesta sin
sombrear un punto, así que se mantiene el valor por defecto y se imprime un aviso; un bake TX lee después esos
campos desde el material compilado en el dispositivo. El espacio de color de una textura se toma del metadato
`colorSpace` de la entrada de fichero (o de `sourceColorSpace`), y los mapas se limitan a `--texture-size` (1024 por
defecto): un mapa es una imagen `float4` en el dispositivo, así que un mapa de 4k ocuparía 268 MB.

## 3.3 El tamaño de una cell

Todo lo que viene después escala con la cell $h$, la longitud en el mundo que representa una Gaussian. La
conversión ofrece cuatro maneras de elegirla, en orden creciente de preferencia.

**Por modelo** (`--density per-model`, el valor por defecto cuando no se da una cámara). $h = R / N$, donde $N$ es
`--resolution` (512 por defecto) y $R$ el mayor de los dos rangos del plano de proyección de la caja del modelo — los
`orthogonalUvs` del original. De ello se siguen dos defectos: un suelo de 16 unidades diluye el coche apoyado en él,
y un panel orientado hacia delante recorre una cell 2.5 veces más fina que el capó, porque su plano de proyección
es el lado corto del coche.

**Por mesh** (`--density per-mesh`). $h = L_m / N$ con $L_m$ el lado más largo de la caja propia del mesh, el
mismo para todos sus triángulos (`cellByLongest`), acotado a $[h_{\min}, h_{\max}]$ dados por
`--cell-min`/`--cell-max`; si se dejan a 0 se derivan de la cell del modelo $h_M$: $h_{\max} = h_M$,
$h_{\min} = h_M/8$, una razón de densidad de 64 entre la parte más fina y la más gruesa. Medido en
`MustangHero.usda` (el coche sobre un suelo de 16 unidades) a `--resolution 512`:

| modo | total | parachoques `Object_199` | capó `Kapoot` | suelo |
|---|---|---|---|---|
| por modelo | 375 199 | 762 (cell 0.0313) | 2 326 | 262 144 |
| por mesh, por defecto | 934 338 | 3 082 (cell 0.0039) | 20 279 | 262 144 |
| por mesh, `--cell-min` = `--cell-max` = 0.03125 | 375 157 | 764 | 2 339 | 262 144 |

La fila de control reproduce el modo por modelo al 0.01 %; la diferencia restante es solo el efecto de la orientación.

**Desde una cámara** (`--cell-from-camera CAM --camera-pixels P`, a partir de Mesh2GS, arXiv 2606.21898). La cell de
cada pieza es lo que cubre un píxel de la cámara `CAM`, de `P` píxeles de ancho, donde la caja del mesh de la
pieza le queda más cerca:

$$
h = \frac{z\,(A/P)}{f}, \qquad z = \max\bigl(\lVert \operatorname{clamp}(\mathbf e, \mathbf b_{\min}, \mathbf b_{\max}) - \mathbf e \rVert,\; z_{\text{near}}\bigr),
$$

con $A$ la apertura horizontal de la cámara, $f$ su distancia focal, $\mathbf e$ el ojo, y el resultado acotado
solo por `--cell-min`/`--cell-max` explícitos. Mesh2GS toma $z_{\min}$ sobre la caja dentro del frustum y sobre las
muestras temporales de la cámara; la distancia euclídea en `--time` es la misma para una caja a la vista y más fina,
nunca más gruesa, para una al lado del frustum. Con dos tarjetas iguales a tres y siete unidades de una cámara de 64
píxeles, la tarjeta cercana recibió 930 Gaussians y la lejana 169: una razón de 5.5 frente a los $(7/3)^2 = 5.4$
esperados. Este es el modo usado en todos los bakes de producción del Corvette (`--cell-from-camera
/World/Camera --camera-pixels` 640, 1280 o hasta 5120).

**El presupuesto.** `--max-splats` (2 M por defecto) se reparte sobre todo el stage. Cada pieza se cuenta primero
(una ejecución del efecto con sitio para una Gaussian, cuyo recuento y exploración cubren todo lo que escribiría una
ejecución); donde el total $W$ supera el presupuesto $B$, la pieza $i$ que quiere $w_i$ recibe
$s_i = \max(1, B w_i / W)$ y recorre una cell $\sqrt{w_i / s_i}$ veces más gruesa. Una cell $k$ veces más gruesa
cubre la superficie con $k^2$ Gaussians menos, así que las cuotas proporcionales y el engrosamiento uniforme son lo
mismo. Con dos tarjetas que quieren unas 1 300 bajo un presupuesto de 800, las cuotas salieron 356 y 361; gastado en
el orden de los meshes, la segunda tarjeta tenía unas 145.

**Límites por triángulo y por ejecución.** Un triángulo recorre como mucho `--max-cells` cells de su caja proyectada
($2^{18}$ por defecto); las que sobran se cuentan y se avisan. A 4736 px, `Infinite_Black` del Corvette perdió así
359 628 cells (agujeros visibles) y se volvió a hacer su bake con `--max-cells 1048576`. La imagen de salida de una
ejecución está limitada a 2 M Gaussians (`kRunCeiling`, 192 MB); una pieza que quiere más se convierte por tramos —
el efecto informa del primer triángulo que el techo cortó y el host vuelve a ejecutar desde ahí con `firstTriangle` —,
lo que da el mismo array que habría escrito una sola ejecución.

## 3.4 Una Gaussian por cell

**La proyección.** Para un triángulo $(\mathbf a, \mathbf b, \mathbf c)$ de normal geométrica $\hat{\mathbf n}_g$,
los dos ejes $(i, j)$ hacia los que menos apunta $\hat{\mathbf n}_g$ definen la proyección; un punto se lleva a
$\mathbf o(\mathbf x) = (x_i - b_{\min,i},\, x_j - b_{\min,j}) / R$, con $R$ el rango en uso (el del modelo, el lado
más largo del mesh o la caja derivada de la cámara). Una cell es $1/N$ de ese cuadrado unidad, es decir, $h = R/N$
en el mundo, acotada como en la sección 3.3. Se emite una Gaussian en cada cell cuyo centro cae dentro del
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
siguen el código original: el jacobiano es sobre las coordenadas triplanares, no sobre las coordenadas de textura; el
factor $1/N$ lo aplica el exportador del original, no ningún shader (sin él cada Gaussian tiene el tamaño del
modelo); y el frame local no son las columnas del jacobiano sino la arista más larga del triángulo $\hat{\mathbf t}$,
la normal de sombreado interpolada $\hat{\mathbf n} = \operatorname{normalize}(\sum_k \lambda_k \mathbf n_k)$ y su
producto vectorial. La covarianza es, por tanto,

$$
\Sigma = R_{\text{frame}}\,\operatorname{diag}(s_u^2, s_v^2, s_n^2)\,R_{\text{frame}}^{\!\top}, \qquad
R_{\text{frame}} = [\hat{\mathbf t}_\perp,\; \hat{\mathbf n}\times\hat{\mathbf t}_\perp,\; \hat{\mathbf n}],
$$

con $\hat{\mathbf t}_\perp$ la arista hecha ortogonal a $\hat{\mathbf n}$. Sin la cota $\ell_{\max}$, 1 859 astillas
de un BMW, vistas casi de canto por su propia proyección, producen columnas del jacobiano veinte órdenes de magnitud
demasiado largas y dibujan púas blancas de metros de largo.

**Anchura.** El $\sigma = 0.65$ del original deja una superficie convertida transparente en torno a un 30 % para un
ray tracer. Medido en un cuadrado unidad iluminado de frente (píxel del mesh 0.5032):

| $\sigma$ | coverage | píxel |
|---|---|---|
| 0.65 | 0.702 | 0.351 |
| 0.85 | 0.848 | 0.425 |
| 1.0 | 0.911 | 0.456 |
| 1.2 | 0.957 | 0.479 |
| 1.5 | 0.986 | 0.493 |

El valor por defecto es, por tanto, $\sigma = 1.0$; `--sigma 0.65` recupera el original.

**Planitud y surfels.** El original escribe $10^{-7}$ unidades del modelo como tercer tamaño, lo que hace depender
la delgadez de una Gaussian de la escala del modelo: un peón de 66 mm se traza bien mientras que un zorro de 100
unidades se convierte en un fantasma bajo el ray tracer. El tercer tamaño es aquí la fracción $\varphi$ (0.1
por defecto) del menor de los tamaños en el plano. `--flatness 0` escribe surfels — Gaussians con una escala
exactamente cero, el disco del Gaussian splatting 2D —, que el rasterizer dibuja mediante una intersección
rayo–plano por píxel y que el path tracer acota a $10^{-4}$ del eje más ancho.

**La opacidad como coverage.** Un punto de una superficie convertida queda bajo varias Gaussians a la vez, así que
una opacidad escrita directamente en cada una cubre de más: una máscara de 0.5 cubría el 96 % de lo que había detrás.
Toda opacidad que lee la conversión (`--opacity`, la constante del material, el valor de un mapa, lo que conserva un
vidrio) se trata como la *coverage* $F$ de la superficie y se convierte en el $\alpha$ de cada Gaussian
desarrollando la transmitancia de la pila hasta segundo orden: sobre la retícula de cells los footprints suman
$2\pi\sigma^2$ y sus cuadrados $\pi\sigma^2$, de modo que

$$
\tfrac{\pi\sigma^2}{2}\,\alpha^2 + 2\pi\sigma^2\,\alpha = -\ln(1 - F)
\;\;\Longrightarrow\;\;
\alpha = \frac{\sqrt{4S^2 + 2S\,L} - 2S}{S} + \tfrac{1}{255}, \quad S = \pi\sigma^2,\; L = -\ln(1-F),
$$

donde el $1/255$ devuelve el área que pierde una Gaussian al cortarse en $1/255$; $F \ge 0.999$ se lleva a 1 y 0 a
0. Medido en planos de coverage constante 0.25 / 0.5 / 0.75, el rasterizer da 0.251 / 0.501 / 0.752 y el ray
tracer lo mismo a tamaño completo, frente a 0.253 / 0.510 / 0.770 con solo el término de primer orden. Un mapa de
recorte por debajo de `--opacity-cut` (0.5) no escribe ninguna Gaussian, así que el presupuesto va donde está la
superficie.

**Un triángulo que no atrapa ninguna cell.** Un triángulo más pequeño que una cell puede no contener ningún centro
de cell; rasterizado, desaparece. Sin un recurso alternativo, eso elimina la grille y las gomas de las ventanas del
Buick (80 137 triángulos se convierten en 34 011 Gaussians). Por ello, tal triángulo recibe una Gaussian en su
baricentro, dimensionada por sus propios segundos momentos (`m2sOwnSize`): la covarianza del triángulo uniforme
$\Sigma_\triangle = \tfrac{1}{12}\sum_k \mathbf d_k \mathbf d_k^\top$ (vértices respecto al baricentro, en el plano
tangente), ensanchada como se ensancha una cell, $\Sigma = 12\sigma^2 \Sigma_\triangle$, con el eje ancho acotado a
$[0.05\,s_{\max}, s_{\max}]$ del tamaño de la cell y el estrecho a al menos 0.02 del ancho. Sin ello, el labio
enrollado de un panel — una tira de triángulos menores que la cell — dibuja una banda de cuentas del ancho de una
cell, de tres píxeles de grosor, a lo largo de cada arista del Corvette donde el path tracer dibuja una línea
finísima. En los bakes del Corvette, el vidrio y los neumáticos salen a aproximadamente una Gaussian por triángulo
tanto a 640 como a 1280 px.

**Orden determinista.** El original añade con un contador atómico, así que dos conversiones del peón producen las
mismas 729 073 Gaussians en órdenes distintos. La emisión son aquí tres pasadas — contar las cells cubiertas por
cada triángulo, una exploración exclusiva en un grupo de trabajo de 256 hilos, escribir las Gaussians de cada
triángulo en su desplazamiento — que comparten una función (`m2sTriangleOf`), de modo que contar y escribir no
pueden discrepar. Dos conversiones son idénticas byte a byte, un presupuesto conserva las primeras Gaussians en el
orden del mesh y la Gaussian $i$ es la misma Gaussian de una pose a la siguiente. Un registro que se decodifica en
nada se escribe como una ranura vacía en lugar de descartarse, de modo que los índices de los arrays siguen
alineados.

## 3.5 Donde una superficie termina o forma un fold

Una Gaussian mide una cell de ancho, y en el Corvette una cell es un píxel de la cámara de conversión. Una parte
más estrecha que eso — la pestaña que un capó dobla hacia abajo dentro de su junta — seguía recibiendo Gaussians
del ancho de una cell en su propio plano; asomaban fuera de la junta sobre la pintura de al lado, llevando la
oscuridad del bake de la junta, y cada borde de panel se convertía en una línea negra gruesa y ondulada donde el
path tracer dibuja una fina.

**Aristas duras** (`mesh2splat_edges.slang`). Por mesh, sobre todas sus piezas, cada arista y cada vértice se
indexan por la posición de sus puntos y se ordenan por radix en el dispositivo; cada tramo de claves de arista
iguales es una arista. Una arista es *dura* cuando está abierta (la usa un solo triángulo: la parte termina o empieza
otro mesh), cuando los dos triángulos que la comparten forman un fold de más de $40^\circ$ (`kCreaseDegrees`) o
cuando la comparten más de dos triángulos (no variedad). Como las claves son posiciones sobre todo el mesh, la
frontera entre dos materiales sobre una misma superficie suave no es una arista. Cada arista dura se convierte en una
recta con un interior, $(\hat{\mathbf m}, c)$ con interior $\hat{\mathbf m}\cdot\mathbf x \ge c$ — el plano que pasa
por la arista, ortogonal a su triángulo, orientado hacia el tercer vértice —, que su propio triángulo conoce como un
bit y que se comunica a cada triángulo que toca cualquiera de sus extremos (hasta cinco rectas cada uno), puesto que
una Gaussian cerca de un vértice puede quedar junto a la arista dura del triángulo vecino.

**Contener el alcance** (`m2sHoldEdges`, `--edge-reach` $\rho$, 1.5 por defecto). Para cada recta contenida en el
plano de la Gaussian, sea $\mathbf m$ su normal en la base tangente, $C$ la covarianza en el plano,
$r^2 = \mathbf m^\top C\,\mathbf m$ la varianza a través de la recta y $d$ la distancia con signo del centro a ella.
El tamaño permitido a través de la recta es $a = \max(d/\rho,\; 0.05\,r)$, y donde $r > a$

$$
C \leftarrow C - \kappa\,(C\mathbf m)(C\mathbf m)^\top, \qquad \kappa = \frac{1 - a^2/r^2}{r^2},
$$

lo que hace $\mathbf m^\top C\,\mathbf m = a^2$ exactamente y conserva la varianza a lo largo de la dirección
conjugada de la recta, de modo que una tira de Gaussians sigue cubriendo la tira. Se ignoran las rectas que no están
en el plano de la Gaussian (la pestaña vista desde la pintura de la que se dobla) y aquellas tras las que el centro
queda a más de un cuarto de su tamaño; las rectas se aplican en un orden canónico, de modo que el resultado no
depende del orden en que se encontraron. La nueva covarianza se rediagonaliza en una tangente y dos tamaños. El
número de Gaussians no cambia; solo su forma.

**Cáscaras cerradas.** La misma pasada cuenta las aristas no compartidas exactamente por dos triángulos, un contador
leído por mesh. Un mesh se marca como cerrado cuando

$$
n_{\text{open}} < 2\sqrt{1.5\,T},
$$

con $T$ sus triángulos ($1.5T$ aproxima sus aristas soldadas), y sus Gaussians llevan
`primvars:athenea:splat:closed = 1`, lo que permite a un player o a un cut descartar nodos que miran hacia
atrás. Una proporción fija de aristas (2 %) no basta: el borde de una lámina abierta crece como la raíz cuadrada de
su área, así que un tubo lo bastante fino pasa por cerrado — con esa regla los cilindros de las llantas (1 %) se
descartan y se ve luz entre los radios. Frente a la raíz cuadrada, las puertas (ninguna arista
abierta), los logotipos (1.4) y las pinzas (1.5) siguen cerrados y las llantas (4.0), los neumáticos (3.1), el capó
(3.1) y el parachoques (4.4) no: 102 de los 192 meshes del Corvette. La puerta de un coche son dos pieles; dibujada
sin descarte, la piel interior se ordena delante de la exterior a parches, una red de grietas oscuras a tamaños
de pantalla pequeños.

**Medido.** La base TX del Corvette a 640 px de `/World/Camera`, con un nuevo bake con las aristas contenidas (1.5 por
defecto) y, como control, con `--edge-reach 0`, ambas de 5 380 804 Gaussians; frame diurno a 1920 × 1080 frente
al path trace del mesh:

| bake | relMSE (HDR) | PSNR (pantalla) | SSIM |
|---|---|---|---|
| base anterior (build anterior) | 0.0900 | 21.83 dB | 0.8555 |
| control, aristas sin contener | 0.0902 | 21.86 dB | 0.8554 |
| aristas contenidas, $\rho = 1.5$ | 0.0887 | 21.93 dB | 0.8567 |

La ganancia global es pequeña porque las aristas son una fracción pequeña del frame; el efecto visible está en
las juntas de los paneles.

## 3.6 Captura del material

Cada registro de Gaussian lleva la posición, la opacidad derivada de la coverage, tres tamaños, una rotación y un
color, seguidos — según lo que necesite la nube — de la normal de sombreado y el metálico, la roughness, la
transmisión y las coordenadas de textura; cuatro joints y sus gradientes; tres entradas de relieve para el
desplazamiento; una entrada de emisión; cuatro entradas de capas; y el operador de forma.

**Base.** El color es el color base del material por su peso base, por el mapa de albedo donde lo haya (decodificado
a lineal). El metálico y la roughness son las constantes del material, multiplicadas por los canales del mapa
empaquetado donde lo haya (azul metálico, verde roughness), como glTF multiplica sus factores; un canal ausente es un
factor uno. El original toma por defecto $(0.1, 0.5)$ solo del mapa, así que todo material sin mapa se convierte
en el mismo plástico; pasando los valores propios del material, un cuadrado convertido bajo una luz de frente da
0.5009 frente a 0.5032 del mesh.

**Normal de sombreado.** Un mapa de normales se lee en el frame local que dan sus UV — $\partial\mathbf p/\partial u$
a partir de las posiciones y las UV del triángulo, ortonormalizado frente a la normal, como hace el shader del mesh
— y el resultado se guarda como normal de sombreado de la Gaussian (`primvars:athenea:splat:normal`) junto a su
frame geométrico. Leído respecto a la arista más larga, cada faceta del mármol del peón gira sus vetas según su
propio ángulo. El disco en sí no se gira salvo que se pase `--normal-map-turns`, lo que abriría la superficie donde
el relieve es pronunciado.

**Capas.** Una Gaussian lleva el peso, el color y el índice especulares; el peso, la roughness y el índice del
clear coat; el color del sheen por su peso y su roughness; y el interruptor de oscurecimiento del coat — en las
unidades de OpenPBR, tres palabras por splat en el dispositivo. Hasta tres mapas sobre estas entradas se muestrean
por Gaussian (`m2sLayersAt`). Es necesario porque la pintura del Corvette (`Car_Paint_Main`) es OpenPBR de
metalicidad 1 sobre una base de aproximadamente 0.05 bajo un clear coat de peso 1: todo lo que el ojo ve de ella es
el coat, y sin el coat un coche convertido sale casi negro.

**Metales de Schlick.** Los metales de OpenPBR y glTF son el `generalized_schlick_bsdf` de MaterialX (el color de
frente, el color especular a $82^\circ$), no un conductor del índice artístico; para un metal oscuro ambos difieren
en un factor dos a ángulos rasantes (a $60^\circ$: 0.172 el conductor, 0.077 el Schlick desde $F_0 = 0.047$). El tipo
de material se escribe por Gaussian (`primvars:athenea:splat:schlickMetal`) y el renderer lo refleja con el
ajuste DFG de sus dos extremos. Tratar en el bake todo metal OpenPBR como polish dieléctrico tiene un coste medido:
el parachoques delantero de un Mustang guarda en sus 27 016 Gaussians la constante que se decodifica en negro;
conservando como cuerpo los lóbulos de Schlick con $F_0 > 0.2$, el color medio del parachoques es
(0.45, 0.51, 0.61).

**Emisión y transmisión.** La emisión es el color del material por su mapa, lineal y sin acotar, en una entrada
propia. La transmisión $T$ y el color de transmisión $\mathbf c_T$ son un canal que el original no tiene (el vidrio
sale blanco opaco). El color se lleva hacia el color de transmisión,

$$
\mathbf c \leftarrow \operatorname{lerp}\!\bigl(\mathbf c,\; \operatorname{lerp}(\mathbf 1, \mathbf c, w_T)\,\mathbf c_T,\; T\bigr),
$$

con $w_T = 1$ para glTF (transmisión teñida por el color base) y 0 para OpenPBR y standard_surface. La opacidad es la
correspondencia de coverage de la sección 3.4 aplicada a $o\cdot o_{\text{mat}}\cdot\text{máscara}\cdot
\operatorname{lerp}(1, g, T)$, siendo $g$ lo que aún cubre una superficie totalmente transmisora (sección 3.8).

**La marca de roughness del material.** La roughness guardada puede ensancharse por el filtro especular (sección
3.7). La roughness propia del material antes del ensanchamiento se escribe aparte,
`primvars:athenea:splat:materialRoughness` (−1 bajo un mapa de roughness, que se lee como nítida). El player la
usa para decidir si el lóbulo de polish responde su parte ocluida desde el campo reflejado del bake (roughness del
material de al menos 0.3: pintura, 0.34) o la deja oscura (cromo 0, metal rugoso 0.22); en el árbol diezmado se
empaqueta en la palabra `pbr` (bit 26 presente, bits 27–31 el valor en cinco bits).

## 3.7 El filtro especular (ensanchamiento de Toksvig)

Una Gaussian tiene una sola normal. Donde una tira de cromo se curva bajo ella, el píxel del path trace promedia un
reflejo que se desplaza por el footprint y muestra una línea fina; la Gaussian muestra la luz que encuentra su única
normal, a pleno brillo, sobre todo su footprint. En el Corvette, sin filtro, el embellecedor de la grille — de cuatro
píxeles de alto — sale como una fila de cuentas blancas, y el cromo da 1.6 veces el path trace mientras que una
bola de cromo se valida dentro del 1 %.

La conversión ensancha por ello la roughness según la dispersión de normales bajo la Gaussian (`m2sSpread`,
`m2sWidened`). La tasa de giro es el mayor ángulo entre normales de vértice por unidad de distancia entre vértices,
sobre las tres aristas del triángulo, y la dispersión es esa tasa por el mayor tamaño de la Gaussian:

$$
s = k\,\max_{e}\frac{\angle(\mathbf n_{e_0}, \mathbf n_{e_1})}{\lVert \mathbf p_{e_1} - \mathbf p_{e_0} \rVert}\,\max(s_u, s_v),
$$

con $k$ = `--specular-filter` (1 con `--transfer`, 0 en otro caso). Con la roughness perceptual $r$ y el
$\alpha = r^2$ de GGX, la dispersión se suma a la varianza de las pendientes (Toksvig):

$$
\alpha'^2 = \alpha^2 + 2 s^2, \qquad r' = \operatorname{saturate}\!\left((r^4 + 2 s^2)^{1/4}\right).
$$

Un triángulo plano, o uno cuyos vértices comparten normal, queda igual. La roughness del coat se ensancha del mismo
modo salvo que la nube lleve el operador de forma.

**El operador de forma.** Para un transfer, la conversión escribe también, en los ejes propios de la Gaussian
$(\mathbf u, \mathbf v)$, el operador de forma simétrico $S$ con $\mathrm d\mathbf n = S\,\mathrm d\mathbf x$,
resuelto a partir de las diferencias de normales de los vértices frente a sus diferencias de posición (`m2sShape`,
guardado como $(S_{uu}, S_{uv}, S_{vv})$). El renderer lo usa para girar el reflejo a través del footprint píxel a
píxel (el «sharp lobe» por píxel). Ensanchar también el coat difuminaría dos veces, así que donde se escribe $S$
el coat no se ensancha, y el player lee los sharp lobes con la roughness sin ensanchar
$r_0 = \max(r'^4 - 2s^2, 0)^{1/4}$ (`unwidened`), con $s$ guardada por Gaussian.

**Medido.** A 640 px el coat salía con roughness 0.17–0.24 y un 5–14 % de paso alto en los folds; a 1280 px la
dispersión bajo cada Gaussian es menor y la roughness del coat sale exactamente 0, así que el ensanchamiento
importa menos cuanto más fino es el bake. Aun así, quitarlo es peor (sección 3.11): con `--specular-filter 0`
aparece una línea oscura discontinua en el fold del faldón y la tira de cromo centellea.

## 3.8 Vidrio

Una Gaussian no puede refractar, pero puede dejar pasar lo que tiene detrás mediante su opacidad, reflejar con su
lóbulo especular y llevar un índice de refracción según el cual el renderer desvía
(`primvars:athenea:splat:ior`, uno por nube; con capas, cada Gaussian desvía según su propio índice especular). La
tarea de la conversión es decidir, para cada pieza transmisora, si el vidrio es un *sólido* (un cuerpo: una bola, una
lente) o una *pared fina* (una lámina o placa cuyas dos caras no desvían nada), porque ambos se dibujan de forma
distinta.

**Clasificación** (`classifySheets`, `mesh2splat_sheet.slang`), para cada pieza transmisora cuyo material no se
declara ya de pared fina. En el dispositivo, las aristas de la pieza se indexan por posición, se ordenan y se cuentan
sus tramos; se suman el volumen con signo $V$ (tetraedros al origen) y el área $A$. Entonces

- *abierta*: más de $1/200$ de las aristas tienen un solo triángulo, o alguna arista tiene más de dos;
- *placa*: cerrada, y el grosor $2V/A$ está por debajo de cuatro cells del modelo o de $0.02\sqrt A$
  ($2V/A$ es la separación de una placa y dos tercios del radio de una bola).

Una pieza abierta o placa se lee como pared fina — salvo que esté *tintada* (luminancia del color de transmisión por
debajo de 0.9), que de momento sigue siendo sólida, ya que una pared fina deja pasar el fondo en gris. El vidrio del
Corvette consiste en placas cerradas de 3–4 mm de grosor, que la regla de aristas por sí sola llama sólidas; la regla
del grosor las reconoce. Cada decisión se imprime, p. ej. para la lente del piloto trasero: «tinted, kept solid
glass: 200 of its 17284 edges open, 0 shared by more than two; 1.199 thick (2V/A) against a cell of 0.007921».
`--thin-glass MAT` y `--solid-glass MAT` la anulan por material.

**Coverage.** Un vidrio sólido cubre `--glass-opacity` de lo que tiene detrás (0.6 por defecto; una lente TX responde
su fondo solo a partir de las cúpulas, así que un vidrio que cubriera por completo ocultaría cualquier otra luz que lo
atraviese). Una pared fina cubre exactamente lo que la lámina refleja de frente, ambas caras con interreflexión:

$$
g_{\text{thin}} = \min\!\left(\frac{2R}{1+R},\; 0.998\right), \qquad R = \left(\frac{n-1}{n+1}\right)^2,
$$

y el frame escala su reflejo de vuelta por $1/R$. Con el índice del parabrisas, 1.16, eso es una opacidad de
0.0055, y el rasterizer dibuja estas láminas con una opacidad de al menos 0.1 y el color reducido en el mismo
factor, lo que mantiene intacto el producto $\alpha\,\mathbf c$ — el reflejo — (relMSE del parabrisas 2.26 → 0.89 bajo
el cielo del taller).

## 3.9 Shadow catcher y light catcher

El suelo no se convierte como superficie; lo que se convierte es su interacción con el objeto, de modo que la
reproducción es solo de Gaussians.

**Shadow catcher** (`--shadow-catcher --prim OBJETO`). El suelo se encuentra automáticamente (el mayor mesh plano
fuera del objeto cuya parte superior está en la base del objeto; `--catcher-ground` nombra uno). Un parche de
cuadriláteros — el footprint del objeto ensanchado `--catcher-margin` alturas (1.5), dividido en una retícula para que
ningún triángulo supere `--max-cells` — se añade a una copia en memoria del stage y se convierte solo, con una cell de
una centésima de la altura del objeto, como un transfer TX cuyos rayos encuentran el objeto y el suelo. Al
dibujar, la opacidad de la Gaussian es $-\ln(\text{razón})/2\pi$, donde la razón es la luz que le llega con el objeto
frente a sin él, ya que una retícula de Gaussians de $\sigma = 1$ cell suma $2\pi$ y una capa de opacidad $a$ deja
pasar $e^{-2\pi a}$. El del Corvette: un parche de 5.9 × 8.3 m, 11 × 11 cuadriláteros, 309 k Gaussians con una cell
de 1.26 cm, con un bake de 79 s. Frente al path trace:

| | bajo el coche (GT 0.182) | contacto del parachoques (GT 0.043) | suelo abierto (GT 0.217) |
|---|---|---|---|
| mapa de sombras propio del suelo de mesh | 0.182 | 0.041 | 0.215 |
| shadow catcher, profundidad 2.4, dos triángulos | 0.144 | 0.002 | 0.206 |
| shadow catcher, profundidad $2\pi$, retícula | 0.192 | 0.030 | 0.220 |

**Light catcher** (`--light-catcher`). El mismo parche, alcanzando `--light-reach` longitudes del objeto (3) por
delante y por detrás, con una cell de una treintava parte de la altura, convertido como un bake de radiancia por
grupo de luces con la cúpula en negro; dibujado de forma aditiva, lleva la mancha de la luz de cruce y el resplandor de
los pilotos traseros.

## 3.10 Validación frente al path trace

`--validate DIR` mide una conversión material a material frente al path trace del stage, para cualquier stage.
Los grupos son los bindings de material del stage (un mesh de dos materiales está en ambos); la conversión es la
del propio comando, ejecutada una vez por grupo con todos los meshes ajenos a él ocultos, de modo que cada opción es
lo que se mide. La referencia se traza una vez (`--validate-paths` 512, `--validate-bounces` 6) y se conserva; el
stage se rasteriza una vez como meshes con su Cryptomatte, y la máscara de cada grupo es la coverage de cada rango
cuyo id es el hash de uno de sus meshes. Dentro de la caja de la máscara, el efecto Measure da relMSE, p99 y las
medias de la nube y del rasterizado del mesh a su lado, escritos en `validate.json`. `--validate-sky white` (o una
imagen) dibuja todo bajo otro cielo con las demás luces del stage apagadas, de modo que el error de un transfer
es solo el de su oclusión y su reflexión. Entre los defectos que mide: una bola de vidrio sólido trazada a la mitad
del brillo del path trace (0.096 frente a 0.195) cuando los rayos no se desvían hacia su interior, y una bola de
pintura bajo una cúpula blanca que da un 13 % por debajo a causa del Fresnel del coat.

## 3.11 Resultados medidos sobre el Corvette

Todos los bakes siguientes son transfers TX, `--cell-from-camera /World/Camera`, 64 caminos por Gaussian,
3 rebotes, sin caminos adaptativos extra, con el filtro especular y el alcance de aristas en sus valores por defecto
(1 y 1.5). «Car crop» es el PSNR de pantalla sobre el recorte de 900 × 420 del coche, y «frame» sobre el
frame completo de 1920 × 1080, ambos frente al path trace del mesh, tras el cut del merge tree a
0.25 px con back-face culling.

**640 → 1280 px.** Las leaves del bake de 640 px medían 3–4 mm en el capó, y las rayas de la puerta y los pelos
del faldón — ausentes en el path trace — estaban en la nube. El coche se vuelve a hacer bake a 1280 px en 8 grupos
de partes de primer nivel equilibrados por recuento, cada uno convertido con todas las demás partes ocultas (nombres
de parte tomados del manifiesto de Cryptomatte). Los ocho bakes escribieron 891 387, 896 496, 809 073, 790 403,
1 631 765, 1 073 954, 1 302 878 y 1 307 473 Gaussians (8 703 429 en total) en 212–304 s de transfer cada uno,
unos 34 min en conjunto.

| | 640 px | 1280 px |
|---|---|---|
| leaves | 5.38 M | 8.70 M |
| capó / puerta | 50 k / 79 k | 151 k / 266 k |
| vidrio, neumáticos | aprox. una por triángulo | aprox. una por triángulo |
| car crop / frame (PSNR) | 16.55 / 22.18 dB | 17.88 / 23.90 dB |

![Figura 6. Car crop (900 × 420 del frame de 1920 × 1080 desde `/World/Camera`): path trace del mesh (arriba), el bake de 640 px (centro) y el bake de 1280 px (abajo), ambos dibujados como el cut de 0.25 px con back-face culling. PSNR del car crop 16.55 dB y 17.88 dB.](figures/img-car-640-1280.jpg)

![Figura 7. El frame completo de 1920 × 1080 del bake de 1280 px, cut de 0.25 px con back-face culling (PSNR del frame 23.90 dB frente al path trace).](figures/img-car-1280-frame.jpg)

Análisis de la nube de 1280 px frente a la de 640 px: los reflejos del capó pasan de 3.2 a 2.0 px de su posición en el
path trace, el contraste de los reflejos abiertos de la puerta de 0.35/0.53 a 0.56/0.74 del path trace, los
agujeros de la grille en el cut grueso de 2.24 a 1.04, y los pelos del faldón de 987 a 459 px solo por la
resolución (figuras 6 y 7).

**Ajustes para 1280 px (variantes A/B/C).** Para elegir los ajustes de 1280 px se hizo bake de ocho partes (ambas
puertas, ambos paneles del faldón, la aleta delantera, la grille cromada, el marco de la grille y el capó) con todo el
coche proyectando sombra, 931 017 Gaussians en cada variante:

| variante | ajustes | resultado |
|---|---|---|
| **A** | como el bake de 640 px ($k = 1$, $\rho = 1.5$) | elegida: sin pelos, la puerta lisa |
| B | `--specular-filter 0` | línea oscura discontinua en el fold del faldón; la tira de cromo centellea |
| C | B + `--edge-reach 1.0` | alargamiento y pelos idénticos a B |

![Figura 8. Los ajustes de 1280 px en el panel del sill: el lateral del coche como contexto (arriba a la izquierda), variante A (arriba a la derecha), variante B (abajo a la izquierda) y variante C (abajo a la derecha). B y C muestran la línea oscura discontinua sobre el fold.](figures/img-sill-variants.jpg)

Las variantes comprueban si las rayas de la puerta son el ensanchamiento de Toksvig del coat saturándose en los
folds y si los pelos del faldón son las Gaussians alargadas de la pasada de aristas. Ninguna de las dos hipótesis se
sostiene a 1280 px: el ensanchamiento es necesario en los folds y un alcance de aristas más estricto no cambia nada,
así que se usan los ajustes de A (figura 8).

**El close-up level a 5120 px.** Una vista cercana está mucho más próxima que `/World/Camera`,
unos 0.5 mm por píxel de pantalla frente a leaves de 3–4 mm. Una conversión del logotipo y la grille desde una cámara
cercana pasa de 108 527 a 404 594 Gaussians entre 1280 y 5120 px y muestra nítidos la bandera a cuadros, la pajarita y
la malla de la grille. Las 66 partes a las que el bake de 1280 px ya había dado como mucho 1.5 Gaussians por
triángulo (insignias, grilles, faros, pinzas, retrovisores, ruedas, limpiaparabrisas, molduras) se vuelven a
hacer bake a 5120 px en siete grupos; los paneles, a 5120 px o tan finos como mantenga cada parte bajo un tope de
leaves (5 M, rebajado a 4.5 M después de que un árbol de 5.01 M no cupiera en la memoria del dispositivo), con
`--max-cells 1048576` (figura 9):

![Figura 9. El logo desde una cámara cercana: leaves del bake de 1280 px (izquierda) y del close-up level de 5120 px (derecha).](figures/img-logo-1280-5120.jpg)

| grupos | leaves | resolución (px) |
|---|---|---|
| fine0–6 (66 partes detalladas) | 18 610 627 | 5120 |
| puerta | 4 084 170 | 5120 |
| Bottom | 4 897 428 | 3392 |
| Part_3 | 4 370 459 | 4480 |
| Infinite_Black | 4 481 738 | 4736 |
| seis paneles grandes y el parabrisas | 15 994 566 | 5120 |
| pan0–4 (paneles pequeños) | 17 088 152 | 5120 |
| Front_Bumper_001 | 5 008 986 | 4480 |
| **24 árboles** | **74 536 126** | |

Los 24 merge trees contienen 74 328 031 merges, 208 095 raíces y 148 864 157 nodos en 49 634 pages (87.4 GB) y
comparten un codebook de oclusión de 8 182 entradas; la sección 6.9 da los detalles. (La conversión de
Front_Bumper_001 emite 5 009 059 Gaussians; su árbol tiene 5 008 986 leaves, la cifra que usa la tabla.)

## 3.12 Limitaciones

- Una Gaussian tiene una sola normal y el footprint de una cell: un tubo de cromo de pocos píxeles de ancho sigue
  siendo un 38 % más brillante que el path trace en el parachoques delantero, y el labio inferior del
  parachoques sale irregular.
- Los faros no llevan emisión en el bake de transfer; los faros delanteros salen oscuros frente al path trace.
- Un vidrio fino tintado se convierte como sólido, y una nube conserva un único índice salvo que lleve capas.
- La correspondencia de coverage supone el solapamiento $2\pi\sigma^2$ de la retícula de cells; los triángulos
  menores que una cell y las cells desplazadas y divididas se solapan de otra forma y no se corrigen.
- Los instanciadores de puntos anidados solo se expanden en el nivel más interno; el movimiento escrito como
  velocidades se lee en la muestra.
- La memoria del dispositivo limita una parte a unos 5 M de leaves para el merge tree que sigue; las partes
  mayores se convierten a una resolución de cámara más gruesa.
