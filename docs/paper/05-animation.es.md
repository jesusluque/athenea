<!-- Copyright (c) 2026 jesus luque. -->

# 5. Assets animados: skeletons, skinning y visibilidad por partes (el peón y el gorrión)

Esta sección describe cómo athenea convierte un asset USD con rig (UsdSkel) en
una nube de Gaussians 3D que un skeleton sigue moviendo. Trata cómo se
convierte la nube, cómo se posa en la GPU en cada frame, cómo su luz y su
autosombra siguen siendo válidas bajo la deformación, cómo la tratan los
LOD y, como aplicación, cómo se reproduce en un navegador y en
un teléfono. Dos
assets recorren la sección. El gorrión molinero (`SparrowBird.usda`, 609
joints, 71 clips de animación) es el caso de prueba animado. El peón de The
Open Chess Set es el control rígido. Todas las cifras proceden de
`docs/decisions.md`, de los ficheros fuente que se nombran junto a ellas o de
los ficheros de medida del player web (`sparkwebGPU`). Las cifras que no se
midieron se indican como no medidas.

## 5.1 Principio de diseño: llevar el rig, no los frames

Hay dos maneras de animar una nube de Gaussians. La primera guarda la
posición, la orientación y la escala de cada Gaussian en cada frame. La
segunda guarda las Gaussians una vez, en la bind pose, junto con los joints
que mueven a cada una, más una transformación por joint y por frame. athenea
usa la segunda, por tres razones, todas medidas:

1. **Tamaño.** Para un personaje verosímil de 150 k Gaussians en 120 frames,
   los arrays por frame cuestan 960 MB en media precisión (1.2 GB en float).
   El rig (cuatro joints por Gaussian, 60 joints por frame) cuesta 2.9 MB. Se
   midió, tal como se escribe, un tubo sintético (200 anillos × 64 segmentos,
   40 joints, 184 320 Gaussians). Su fichero ocupa 9.27 MB con 25 instantes y
   9.50 MB con 121, frente a unos 713 MB para el mismo rango como arrays por
   frame. Cinco veces los frames cuestan 230 kB.
2. **Exactitud entre muestras.** Un rig es exacto en cualquier instante del
   rango. Una nube muestreada solo es tan buena como la interpolación del
   lector.
3. **Identidad estable.** La nube posada se escribe en buffers que la
   sobreviven, así que un renderer ve la misma nube de un frame al siguiente.
   Eso es lo que permite al ray tracer hacer *refit* de su estructura de
   aceleración en lugar de reconstruirla (§5.6).

El coste de esta elección es que todo lo que se hizo bake en espacio de mundo
(radiancia, transfer en spherical harmonics, visibilidad) deja de ser válido
en cuanto se mueve un miembro. De esto se ocupan §5.7 y §5.8.

## 5.2 Lectura de un stage con rig

El conversor (`athenea mesh2splat`) no pasa por Hydra. Lee los meshes con su
propio `MeshStage`, lo que deja dos maneras de tratar un mesh con skinning:

- **Posado.** Sin `--skinned`, el stage se posa antes de leerlo.
  `UsdSkelBakeSkinning` escribe los puntos posados como muestras de tiempo en
  una capa de override en memoria sobre la capa raíz, sobre
  `GfInterval(t, t)`. Esto cuesta una pose, deja intacto el fichero en disco
  y deja sin tocar un stage sin `SkelRoot`. Prueba: un cuadrado movido por un
  joint que se desliza (1.2, 0.4, 0) entre t = 0 y t = 1 se convierte en unos
  límites, plegados en el device, que difieren exactamente en ese
  deslizamiento.
- **Llevado.** Con `--skinned` se lee la **bind** pose y se omite
  `UsdSkelBakeSkinning`. Posar primero aplicaría el skinning dos veces a la
  geometría. Para cada mesh, `MeshStage` resuelve su skel binding con un
  `UsdSkelCache` sobre cada `SkelRoot` y un `UsdSkelSkinningQuery` por prim
  con skinning. Devuelve las influencias de los joints, `geomBindTransform`,
  la ruta del skeleton y el orden de joints del skeleton.

**Orden de los joints.** Un mesh puede enumerar su propio subconjunto de
joints en `skel:joints`. El mapper de UsdSkel solo traduce del orden del
skeleton al del mesh, así que se aplica sobre la identidad para saber el
índice de skeleton de cada índice de mesh. Las influencias se escriben siempre
en el orden del skeleton, de modo que una nube tiene un solo orden de joints,
sea cual sea el número de meshes de los que se convirtió. Un mesh de un stage
con skinning al que no mueve ningún joint (el stage del gorrión lleva un
cilindro y un plano junto al pájaro) recibe igualmente un registro de
influencias por Gaussian: cuatro joints de peso cero, que el skinner lee como
«quédate en la bind pose». Así el rig y la nube se mantienen alineados uno a
uno.

**`--skinned` fuerza `--no-bake`.** Un bake de radiancia guarda el entorno y
los bounces de la escena (el suelo bajo una pata), y llevarlo hacia arriba con
un miembro levantado equivale a girar un lightmap. Una nube con skinning lleva
su material y se hace relight en cada frame. Su transfer de luz, cuando se
pide, se guarda de una forma que gira con la Gaussian (§5.7).

## 5.3 Influencias por Gaussian, y sus gradientes

`mesh_pack.slang` escribe las cuatro influencias más pesadas de cada esquina
de triángulo en una segunda imagen con exactamente la misma disposición que
la del mesh, para que el efecto AOFX direccione las dos igual. El efecto sabe
dónde está cada Gaussian dentro del triángulo, así que mezcla allí las
esquinas (`blendInfluences`, `mesh2splat.slang`). Con coordenadas
baricéntricas $\lambda_a,\lambda_b,\lambda_c$ y pesos de esquina $w_{cj}$, el
peso bruto del joint $j$ es

$$\tilde w_j = \sum_{c\in\{a,b,c\}} w_{cj}\,\lambda_c .$$

Entran doce pares (joint, peso) y salen cuatro. Un joint ya presente suma el
peso, un hueco vacío recibe un joint nuevo y, si no, el nuevo sustituye al más
ligero. Elegidos los cuatro joints, cada peso conservado se recalcula como su
mezcla completa y los cuatro se renormalizan, $w_k = \tilde w_k / W$ con
$W=\sum_{i\in K}\tilde w_i$. Por tanto, una Gaussian siempre queda llevada por
completo.

**Gradientes.** $\tilde w_j$ es lineal sobre el triángulo, con
$\nabla\tilde w_j=\sum_c w_{cj}\nabla\lambda_c$, donde
$\nabla\lambda_b = (e_2\times n)/|n|^2$,
$\nabla\lambda_c = (n\times e_1)/|n|^2$ y
$\nabla\lambda_a = -\nabla\lambda_b-\nabla\lambda_c$. El gradiente del peso
renormalizado sale de la regla del cociente:

$$\nabla w_k = \frac{\nabla\tilde w_k - w_k\sum_{i\in K}\nabla\tilde w_i}{W}.$$

Es exacto en el centro de la Gaussian. Sobre un disco solo actúan las dos
componentes a lo largo de los ejes de reposo de la Gaussian $(\hat u,\hat v)$,
y los gradientes suman cero, así que solo se guardan los de los tres primeros
joints. Son tres palabras de dos halves, **12 bytes por Gaussian**, en
`primvars:athenea:splat:jointWeightGradients` (`half[]`, `elementSize` 6).
El cuarto gradiente es menos la suma de los otros tres. Con los gradientes, el
registro del efecto para una Gaussian llevada crece a nueve entradas.

**Medido.** Un cuadrado ligado por entero a un joint se convierte en 1056
Gaussians, todas con ese joint y todo el peso. A través del efecto, 272
Gaussians de un quad cuyas esquinas nombran todas el joint 3 salen en el joint
3, con los demás huecos vacíos y los pesos sumando uno. Con un solo joint en
todas partes, el gradiente guardado es menor que $10^{-3}$.

**Truncamiento.** Cuatro influencias es un truncamiento. En el gorrión, el
27 % de los vértices del ala tiene más de cuatro (6686 vértices,
`elementSize` 25), y una Gaussian toma la unión de los joints de tres
esquinas, así que la fracción de Gaussians truncadas es al menos igual de
grande. Esa fracción no se ha medido.

## 5.4 Skinning de una Gaussian en el device

`scene::SplatSkinner` lanza `shaders/athenea/scene/splat_skin.slang`, un hilo
por Gaussian. Sus entradas son la nube de reposo, los pares (joint, peso), las
transformaciones de skinning de los joints en el instante actual y tres
matrices: la transformación de bind $G$, `skelToWorld` y `worldToPrim`. Las
matrices de USD son row-major y actúan sobre vectores fila, así que el host
las traspone como hace `geom::Skinner` con los meshes (figura 14).

![Figura 14. Skinning en el device: la nube en bind pose, los pesos de joint por Gaussian y sus gradientes, y los transforms de joint del instante actual dan la media y la covarianza en pose; las tablas de visibilidad por parte, con bake único en bind pose, dan la sombra de la nube en pose.](figures/fig-skinning.es.svg)

### 5.4.1 La media

Sea $q = G\,p$ el punto de reposo en espacio de bind, $X_j q = A_j q + t_j$ la
transformación de skinning del joint $j$, y
$M=\texttt{worldToPrim}\cdot\texttt{skelToWorld}$. La media posada es linear
blend skinning, con exactamente la aritmética que usa el skinner de meshes:

$$\mu' = M \sum_j w_j\,X_j\,(G\,\mu).$$

### 5.4.2 El marco: el Jacobian entero

La covarianza posada debería ser la covarianza de reposo empujada por la
derivada del mapa de posiciones. Cuando los pesos varían en el espacio, esa
derivada es

$$J = L(M)\Big[\sum_j w_j A_j\,L(G) \;+\; \sum_j (X_j q)\,\nabla_p w_j^{\mathsf T}\Big],$$

donde $L(\cdot)$ es la parte lineal. El primer término es el habitual «mezclar
las partes lineales de los joints». El segundo, el término *elástico*, es el
que estira una Gaussian a través de una flexión donde dos joints no coinciden
en adónde va su punto. Con dos joints separados por un ángulo relativo
$\theta$ y una transición de pesos de longitud $L$, su tamaño relativo al
primer término es de unos $2\tan(\theta/2)\,d/L$, que es del orden de uno en
ángulo recto.

El kernel construye el término elástico en el mismo bucle que forma $X_k q$.
Acumula $\sum_k (X_k q)\,g_k$ sobre cada eje de reposo, *relativo al punto
mezclado*. Los gradientes suman cero, así que es la misma suma, pero se
calcula como diferencia de puntos cercanos, lo que se mantiene exacto en
float lejos del origen. Un joint de peso cero pero con gradiente no nulo
también contribuye.

La covarianza posada en el plano de la Gaussian es

$$\Sigma' = J\,E\,S^2E^{\mathsf T}J^{\mathsf T},\qquad E=[\hat u\ \hat v],\ S=\mathrm{diag}(s_u,s_v).$$

El kernel toma una base ortonormal $(e_1,e_2)$ del plano generado por
$a=J\hat u\,s_u$ y $b=J\hat v\,s_v$, escribe $[a\ b]=[e_1\ e_2]K$ con $K$
triangular superior, y diagonaliza $C=KK^{\mathsf T}$ en forma cerrada. El
autovalor mayor es $m+r$, con $m=\tfrac12(c_{00}+c_{11})$ y
$r=\sqrt{\tfrac14(c_{00}-c_{11})^2+c_{01}^2}$. El menor se calcula como
$(k_{00}k_{11})^2/(m+r)$ en lugar de $m-r$, que en float se cancelaría a nada
para una Gaussian fina. Los nuevos ejes son los autovectores y los nuevos
tamaños las raíces cuadradas de los autovalores. El primer eje es el
autovector más cercano al primer eje llevado, y donde $c_{01}$ es puro
redondeo la base se conserva exactamente, para que una Gaussian redonda no
gire. El tercer eje es la normal del plano, y el tercer tamaño (el grosor de un
disco) queda como se convirtió. La normal de shading gira con el cofactor del
$J$ entero, porque $(Ja)\times(Jb)=\mathrm{cof}(J)(a\times b)$.

El kernel solo escribe la posición, la rotación y las dos escalas del plano.
La opacidad, el color, los harmonics, los canales PBR y la palabra que guarda
el tercer tamaño no se tocan. Un frame cuesta, por tanto, un kernel sobre la
nube, y no se vuelve a decodificar nada.

### 5.4.3 Medido

La prueba de la tira, en `tests/scene/test_loading.cpp`, usa 4096 Gaussians
sobre una rampa lineal de pesos, dobladas un cuarto de vuelta. Compara la
covarianza posada decodificada de cada Gaussian con la analítica
$J=(1-w)I+wR+(Rp-p)\nabla w^{\mathsf T}$, y cuenta los errores relativos de
Frobenius por encima del 3 %. La tolerancia viene del formato: un cuaternión
smallest-three de diez bits no puede fijar un eje con más precisión que unos
$\sqrt2/1023$.

| caso | solo la mezcla, Gram–Schmidt (antes) | Jacobian entero, marco propio (después) |
|---|---|---|
| doblada, ejes a 0.1 rad de la rampa | 2048 de 4096 | 0 |
| doblada, ejes girados 30° | 2048 | 0 |
| giro rígido y desplazamiento (control) | 0 | 0 |
| un joint que cizalla, pesos constantes | 4096 | 0 |
| doblada, sin gradientes guardados | 2048 | 1984 |

Una prueba anterior colocó 4096 Gaussians sobre una hélice, cada una con su
propio marco y tamaño. En reposo, cada Gaussian se queda donde estaba. Bajo
un giro rígido de 1.2 rad alrededor de un eje inclinado más un desplazamiento,
cada Gaussian se lleva rígidamente. Solo se leen cuatro contadores.

En el clip `Flight` del gorrión (61 frames, 1500 triángulos del ala, el mesh
posado como referencia), el término de mezcla solo se aparta del mapa propio
del mesh una mediana del 4–6 %, un percentil 90 del 14–19 % y un 99 del
37–50 %, en la mano del ala y los abanicos de plumas. Con el término elástico,
los percentiles 90 y 99 bajan al 5–6.5 % y al 11–13 %. En imágenes de path
trace (1 125 022 Gaussians, 1024×768, frames 5, 31, 36) los dos kernels
difieren en unos pocos niveles de 8 bits en cerca de una décima parte de los
píxeles (p99 14–16), y las alas se ven iguales. El relMSE de la imagen entera
frente al mesh empeora un 2–3 %, pero una ablación muestra que es cosa de la
métrica: los gradientes negados, intercambiados, a la mitad y al doble
(0.0900, 0.0907, 0.0887, 0.0983 frente a 0.0904) se ordenan por el tamaño del
cambio, no por su corrección, porque domina el error de conversión. La
corrección se apoya en la prueba de la tira y en un diagnóstico sobre 174 305
Gaussians del ala: entre vecinas con los mismos cuatro joints, los gradientes
guardados predicen el cambio de los pesos guardados con un residuo RMS del
12.5 %, frente al 158 % con los ejes intercambiados y el 100 % sin gradiente.

## 5.5 El fichero: el rig como USD

Una nube con skinning es un `ParticleField3DGaussianSplat`. El rig se
escribió primero bajo `AtheneaSplatSkinningAPI`, y ahora se escribe con los
nombres propios de UsdSkel. `mesh2splat --skinned` aplica `SkelBindingAPI` y
escribe `skel:jointIndices` y `skel:jointWeights` (`elementSize` 4),
`skel:geomBindTransform`, `skel:joints` y `skel:skeleton`. También escribe
`primvars:athenea:splat:skinningXforms`: una matriz por joint, en el orden del
skeleton, **muestreada en el tiempo**. Es la única propiedad de una nube
animada que cambia de un frame a otro. Las transformaciones son
`UsdSkelSkeletonQuery::ComputeSkinningTransforms` en cada instante de
`--range START:END[:STEP]`.

**Ligado al Skeleton.** usdSkelImaging construye ext computations solo para
meshes, curvas y puntos. Sí resuelve, en cambio, los `skinningTransforms` de
cualquier Skeleton en cualquier instante. `HdAtheneaSkelSplatSceneIndex`
(`modules/usd/src/SkelSplat.cpp`) pone esas transformaciones resueltas en la
nube como `athenea:splat:skinningXforms`, remapeadas por `UsdSkelAnimMapper`
cuando la nube nombra un subconjunto de joints, y reenvía a sus nubes los
avisos de cambio de un Skeleton. El primvar que escribe la conversión pasa a
ser una **caché**: un stage que compone la nube sin su Skeleton sigue
moviéndose, y donde están los dos gana la caché. El skinning por cuaterniones
duales (`skel:skinningMethod`) no está implementado para nubes; un prim así se
anima por linear blend, con un aviso que se imprime una vez.

**Medido.** 512 Gaussians ligadas a un Skeleton de un joint cuya animación
pone el joint en t = 1 donde la caché lo tiene en t = 2 dibujan, con la caché
bloqueada, el frame t = 2 de la caché en t = 1 píxel a píxel (p99 0, máx. 0).
En el gorrión, los 609 joints que recibe el motor a través del Skeleton
coinciden con los de la capa de rig; las transformaciones propias de USD
frente a la caché escrita por Blender difieren como mucho $5\times10^{-4}$, y
los frames difieren en una media de $8\times10^{-6}$.

**Escala de tiempo.** Sin `timeCodesPerSecond`, la nube se leía a 24 bajo un
stage de 30 fps y se retrasaba respecto al mesh (error medio 0.031 en el
frame 16, 0.049 en el 61); con la tasa escrita, $1.4\times10^{-5}$ y
$1.1\times10^{-5}$.

**Clips.** El gorrión llega como cinco ficheros FBX con 71 clips. Cada clip
solo cambia `skinningXforms` (609 matrices por frame), así que un clip es una
capa sobre una nube compartida: 377 MB para los 71, frente a 22 GB para 71
nubes convertidas. `SparrowClips.usda` los ofrece como un **variant set** de
USD. La nube entra como payload, porque la opinión local de una subcapa
ganaría a la variante, y cada clip entra como referencia dentro de su
variante. El motor sigue siendo genérico: lista los variant sets, selecciona
uno en una capa de override en memoria e informa de `animationRange()`, el
intervalo de las muestras escritas (por ejemplo 1..25 para `land_hop_1`,
1..33 para `air_fly_A0`). Elegir un clip lee unos 3 MB, no 213 MB.

| asset (`mesh2splat --skinned`) | Gaussians | joints | instantes | fichero |
|---|---|---|---|---|
| tubo con rig, resolución 512 | 184 320 | 40 | 25 / 121 | 9.27 / 9.50 MB |
| zorro de Khronos, resolución 512 | 204 517 | 24 | 28 | 13.35 MB |
| gorrión, resolución 1400 (primera conversión) | 4 991 908 | 609 | 33 | — |

## 5.6 El motor: pose, límites, refit y motion blur

`Engine::carryCloud` empareja los dos arrays del fichero en la disposición
(joint, peso), conserva la nube subida como bind pose y ejecuta el skinner
sobre un `GpuSplats` aparte que comparte los buffers de harmonics y PBR, ya
que el skinner no escribe ninguno de los dos. Al medirlo aparecieron dos
errores. Primero, la extensión de una nube posada no es la de la bind pose,
así que ahora se vuelve a plegar en el device después del skinning.
Segundo, `Engine::bounds()` plegaba la nube subida en lugar de la dibujada.

**Un paso de tiempo no es una nube nueva.** Al principio, cada cambio de time
code volvía a decodificar todos los arrays y reconstruía en la CPU los pares
de influencias (137 MB en el gorrión de 4.27 M Gaussians). Ahora los arrays
se comparan por identidad (excepto `skinningXforms`), los pares solo se
reconstruyen al subir, y los arrays sin muestras de tiempo se retienen
(`_held`), porque los arrays enteros comprimidos del fichero crate llegan en
buffers nuevos en cada lectura. Viewer, 400 frames a 1280×720:

| | antes | después |
|---|---|---|
| subidas de la nube en 400 frames | 400 | **1** |
| raster, mediana / en caliente | 134.96 / 31.56 ms | **56.87 / 13.40 ms** |
| `rt`, mediana / en caliente | 268.66 / 55.31 ms | **178.99** / 55.05 ms |

**Refit, no reconstrucción.** Una pose no cambia ni el número de partículas
ni su orden, así que `GaussianRayTracer::sync` hace refit. En la ruta
hardware recalcula los proxies y actualiza en sitio el BLAS de cada trozo. En
la ruta de BVH por compute hace refit del tree una altura cada vez, de abajo
arriba, sin lectura de vuelta. Cada `refitsPerRebuild` = 240 refits se hace
una reconstrucción completa. Frente a reconstruir en cada pose, las imágenes
coinciden con p99 0, máx. 1. Un tree sin refit da p99 255. Gorrión a
resolución 256 (300 862 Gaussians, BVH por compute): 140.9 / 143.1 /
135.6 ms por frame antes, 84.7 / 83.9 / 83.7 ms después (−40 %).

**Motion blur desde el skeleton.** `ParticleField` muestrea `skinningXforms`
a lo largo del shutter. El skinner mezcla la segunda pose en el mismo bucle
(solo los puntos, cerca de un tercio de kernel) y escribe el desplazamiento de
cada Gaussian como dos palabras de halves. El rasterizer convierte ese
desplazamiento en un término de covarianza de rango uno. Hydra toma las
muestras que abarcan el shutter en lugar de recortarlas, así que el
desplazamiento se escala por $(t_c-t_o)/(t_1-t_0)$; sin ese factor las alas
se desenfocan el doble. En el gorrión de 5.9 M a 1280×800, un frame tarda
90.0 ms con shutter 0 y 100.5 ms con shutter 0.5 (+11 %).

## 5.7 Luz que gira con un miembro: lobes zonales de transfer

Una nube estática puede llevar un transfer precalculado de nueve spherical
harmonics en espacio de mundo. Bajo un skeleton esos harmonics conservarían
el cielo de la pose de conversión, y girarlos cuesta una matriz $9\times9$ por
Gaussian. athenea guarda en su lugar uno o dos **lobes zonales** por
Gaussian, con el eje de cada lobe expresado en el marco propio de la
Gaussian. Una función zonal gira al girar su eje, una sola $3\times3$, así
que cualquier marco que produzca el skinner arrastra los lobes consigo, y
no hace falta asignar ninguna Gaussian a una parte.

Un lobe de eje $a$ y coeficientes $z_l$ es, en harmonics,
$g_{lm}=z_l\,k_l\,Y_{lm}(a)$ con $k_l=\sqrt{4\pi/(2l+1)}$. Para un eje fijo,
los mejores coeficientes son la proyección
$z_l = k_l\sum_m f_{lm}Y_{lm}(a)$. Por el teorema de adición, el residuo es

$$\|f-g\|^2=\|f\|^2-\sum_{l=0}^{2} z_l^2,$$

así que el eje se elige para maximizar $\sum_l z_l^2$. En el device, la
búsqueda prueba la dirección de la banda 1, la normal y 64 direcciones de
Fibonacci sobre un hemisferio, y después refina con un paseo que divide su
paso por dos. El segundo lobe se ajusta al residuo del primero, y los dos se
reajustan uno contra otro dos veces (el ajuste ZH de Sloan). Los datos son
diez floats por Gaussian, `primvars:athenea:splat:transferZonal`
(`elementSize` 10): por lobe, el $(u,v)$ octaédrico del eje en el marco y
$z_0,z_1,z_2$. En el device es el buffer `transfer` de siempre con
`transferCount` = 10. Al leer, `splatTransferFrame` lleva cada eje al mundo con
la rotación actual de la Gaussian y reconstruye los nueve harmonics.

**La pose del bake.** Al principio, el bake posaba la nube en `--time` con
cuatro influencias por Gaussian, mientras Hydra animaba el mesh con todas las
suyas (diez por vértice en las plumas, 25 en las alas). Donde no coincidían,
el rayo del bake no encontraba su superficie, y 566 603 de 1 504 160
Gaussians salieron transparentes. Ahora el bake mantiene el skeleton en su
bind pose (`StageRenderer::holdBindPose`) y posa la nube con transformaciones
identidad: 1 504 156 de 1 504 160 encuentran su superficie. El transfer guarda
por tanto la autosombra de la bind pose (alas extendidas). La oclusión que
proyecta otro miembro en otra pose es cosa de §5.8.

**Comprobado.** En `[zonal]`, un transfer de dos lobes sobrevive al ajuste y
al almacenamiento en f16 dentro del 0.5 %. Girar el marco de una Gaussian por
$R$ y leer a lo largo de $Rw$ da lo que la Gaussian sin girar da a lo largo de
$w$. En una prueba USD, 512 Gaussians llevadas por un joint que gira 60°
coinciden con la misma nube bajo un Xform de esa rotación, con relMSE por
debajo de $10^{-4}$ en las rutas raster y trazada. El error del ajuste en el
gorrión no se ha medido. La indirect half del transfer no se guarda zonal.

## 5.8 Campos de visibilidad por partes: bake una vez, un producto de lecturas de tabla por frame

Una nube con relight también tiene que hacerse sombra a sí misma, y las
medidas descartaron las rutas obvias. En la ruta trazada `--splat-shadows` no
cambiaba nada, porque un splat trazado no tenía pasada de sombra. En la ruta
raster costaba 36× el frame (49.58 → 1810.24 ms) y aplastaba el pájaro a
negro. Barrer el desplazamiento propio de 1 a 800 sigmas propias movía la
imagen media de forma monótona del 0 % al 99 % del valor sin sombra, sin
meseta. La literatura coincide en que la visibilidad con bake no sobrevive al
movimiento articulado salvo si se factoriza por parte del cuerpo: Lin et al.
usan MLP direccionales por parte, y DNF-Avatar un grid escalar de AO por
hueso. athenea combina las dos ideas: una **tabla direccional por parte**.

**Partición.** `partitionJoints` toma el subárbol más grande que aún no es una
parte, con al menos `minJoints` (6) joints, y lo convierte en parte, hasta que
hay `--parts` (12). En el gorrión salen raíz, centro, pecho, pelvis, cabeza, y
cada ala en tres segmentos. Cada Gaussian pertenece a la parte de su joint
dominante.

**Bake** (`athenea visibility`, `splat_visibility_bake.slang`). Se hace una
vez, en la bind pose. Para la parte $p$, un grid de $N^3$ sondas cubre la caja
de la nube más un margen del 5 %. Cada sonda guarda un mapa octaédrico de
$O^2$ direcciones. Un texel guarda la transmitancia de un rayo que sale de la
sonda a través solo de las Gaussians de la parte $p$. Las Gaussians de la
parte se reúnen en una nube propia y se trazan con rayos inline, con el mismo
producto de $(1-\alpha)$ que usan los kernels de sombra. Cada sonda guarda
además un término ambiente, su media sobre las direcciones, que se usa para
los domes.

**Lectura** (`splat_visibility_read.slang`). En cada frame, `visPartFrames`
calcula un mapa afín por parte desde el espacio posado de vuelta al espacio
de bind donde vive el campo:

$$x_{\text{rest}} = U A_{k_p}^{-1}\,(x' - t_{k_p}) - U g,\qquad U=L(G)^{-1},$$

donde $k_p$ es el joint cabeza de la parte, $X_{k_p}=(A_{k_p},t_{k_p})$ su
transformación de skinning y $g$ la traslación de $G$; a este mapa se le
llama $T_p^{-1}$. Antes de precalcularlo costaba dos inversas $3\times3$ por
Gaussian, por parte y por luz, unos 260 millones por frame en el gorrión;
precalcularlo bajó la lectura de 12.7 a 8.0 ms (5 887 323 Gaussians,
1600×1000). El factor de la Gaussian $i$ para la dirección de luz
$\omega_\ell$ es entonces el producto sobre las demás partes:

$$V_{i\ell}=\prod_{p\ne \pi(i)} F_p\!\left(T_p^{-1}x_i,\;L(T_p)^{-1}\omega_\ell\right),$$

donde $\pi(i)$ es la parte propia de la Gaussian, $F_p$ se lee de forma
trilineal sobre las sondas y bilineal sobre el mapa octaédrico, y vale 1 fuera
de la caja de la parte. Un dome usa el término ambiente. Los factores van al
mismo `shadowFactors` que leen las dos rutas, así que la ruta trazada, que no
tenía ninguna sombra de splats, también los recibe. No se traza ningún rayo
por frame.

**Hallado al construirlo.** Los texels sin escribir de una parte vacía se
leían como sombra total y dejaban el pájaro negro; ahora un grid de 0 la marca
y se salta. El bake traza rayos inline, que el destino CUDA no tiene, así que
es un módulo aparte y los devices CUDA leen campos con bake hecho en otro
device. Una parte no se hace sombra a sí misma: su campo, leído en una
Gaussian suya, es la multitud en la que esa Gaussian está. Un mapa de
$8\times8$ (unos 22° por texel) no resuelve el borde de un ala; 16 y 32 son
exactos fuera de una banda de una cell de sonda.

**Medido** en el gorrión: 4 269 858 Gaussians, 609 joints, 12 partes, grid 24,
octava 16.

| | |
|---|---|
| bake (M5 Pro) | 22 s, 3 538 944 texels por parte ($24^3\times16^2$) |
| campos en el fichero | 81 MB (dos f16 por palabra) |
| factores por debajo de ½, frame 31, sol + dome | 21.3 % |
| viewer raster, 400 frames, mediana / en caliente | 71.82 / 9.69 ms (56.87 / 13.40 sin ellos) |
| viewer `rt`, mediana / en caliente | 193.88 / 51.65 ms (178.99 / 55.05 sin ellos) |
| L4 (CUDA) raster, 8 frames 1280×720 | 73.04 ms (67.48 sin ellos) |
| L4 (CUDA) `rt` | 128.77 ms (121.48 sin ellos) |

En la L4 los campos cuestan 5–7 ms por frame para
4.27 M Gaussians × 2 luces × 11 partes de consultas. La sombra cae donde cae
la del mesh (el ala lejana, el flanco bajo el ala levantada, la raíz de la
cola) y sigue a las alas de un frame a otro. Con todas las tarjetas de plumas
(9.5 M Gaussians a resolución 1400), el bake se quedó sin memoria en la parte
más grande tanto en el M5 Pro (Metal) como en una L4 a través de Vulkan (un
BLAS de 1.17 GB para unos 3.3 M Gaussians). Por eso el bake del pájaro se hace
a resolución 1100.

**Límites.** Se deja fuera la autosombra local de una parte (pluma sobre
pluma). El término ambiente es una media sobre la esfera, no un lobe coseno.
Se supone que las partes son rígidas y que ocluyen de forma independiente. El
suelo bajo el pájaro se sombrea con un mapa aparte en el espacio de la luz,
que se estabiliza frente a la «respiración» de una caja posada (dimensionado a
partir de la esfera de reposo, encajado en el grid de texels del mundo). Una
alternativa con un grid de densidad no está implementada.

## 5.9 LOD y diezmado de nubes con skinning

El octree de `.athc` hace merge de cells en orden de Morton. Eso no sirve para
una nube con skinning, porque una cell que tomara ala y cuerpo juntos no
sabría a qué joint seguir. Dos mecanismos lo sustituyen.

**Niveles discretos por reconversión.** `mesh2splat --lod-levels N` convierte
a la resolución pedida, después a la mitad, y así sucesivamente, cada nivel
una nube con su propio rig. `Engine::lodLevelsFor` dibuja el nivel más grueso
cuya cell no abarca más de `athenea:lod:threshold` píxeles (1 por defecto) en
el punto más cercano de sus límites *posados*. Los niveles no dibujados no se
posan (`deferredPose`). Clips del gorrión a 1100 / 550 / 275 cells
(5.9 M / 1.5 M / 0.4 M Gaussians), 1600×1000: 49.6 → 51.5 ms cuando el
pájaro llena el frame; **71.3 → 24.7 ms** a cuatro veces la distancia. Todos
los niveles quedan residentes con sus propios campos de visibilidad (81 MB
cada uno), y el cambio de nivel es un salto, no un fundido.

**Diezmado que conserva el rig.** `usd::decimateStage` encuentra cada array
por Gaussian y hace merge de cada uno según su tipo (`lod_attributes.slang`).
Cuando `skel:jointIndices` tiene un `skel:jointWeights` a juego, el par se
combina como rig (modo 3). Sobre el tramo de registros $R$ al que representa
una Gaussian conservada, cada uno con peso de merge $\omega_r$,

$$c_j=\sum_{r\in R}\omega_r\sum_k w_{rk}\,[\,j_{rk}=j\,],$$

reuniendo hasta 16 joints distintos. Se conservan los `width` más pesados y
se renormalizan, $w'_j = \min(c_j/\sum_{\text{kept}} c,\,1)$: una división en
el device no está correctamente redondeada, y un merge entero sobre un
joint salía a $1 + 10^{-7}$. El joint dominante de un
registro se pliega además en la clave de merge, así que un merge nunca cruza
el borde de un joint. Comprobado: un suelo cuyas mitades pertenecen a dos
joints pasa de 4096 a 256 Gaussians, cada Gaussian conservada entera en el
joint de su lado. El nivel más fino del gorrión (5 887 323 Gaussians, con ids,
material, partes de visibilidad y rig) conserva el 87.7 % en 28 s. Posado en
el tiempo 20, se dibuja como la nube sin diezmar con un RMS de imagen de
0.002.

## 5.10 El gorrión

El gorrión y un BMW fueron los dos primeros assets convertidos que no se
crearon para este renderer, y la mayor parte de lo que destapó estaba fuera
de las matemáticas del skinning (figura 15).

- **Exportación de la animación.** El exportador de Blender no escribía
  muestras de joints para este rig importado de FBX, así que
  `scripts/sparrow-anim.py` muestrea la pose evaluada. Resolver un cambio de
  base frente a `restTransforms` destrozaba la cabeza, el pico y los dedos,
  porque reposo y bind son poses distintas en este pájaro (peor 1.989 en
  `Toe.L.010`). Ahora escribe $W(t)=P(t)\,L^{-1}B$, de modo que $W(t)B^{-1}$
  es la propia matriz de skinning de Blender; lo que queda es cuantización,
  $4.9\times10^{-5}$–$7.1\times10^{-5}$ en los frames 1, 31, 61.
- **Stage corregido.** `SparrowBird.usda` lleva los arreglos del asset: nada
  de metálico en plumas ni ojos, un mapa de normales de pluma sin
  premultiplicar (`UV2_treesparrow_normal_fixed.png`) y el segundo juego de
  UV (`UVMap_001`) para el alfa de las plumas. Todas las medidas posteriores lo
  usan.
- **Recortes y budget.** Respetar la `opacity` conectada al alfa de las
  plumas bajó su budget de 15 378 695 a 5 898 220 Gaussians a
  resolución 1400. Los meshes por encima del techo de 2 M por pasada se
  convierten por trozos; con todas las tarjetas, el pájaro tiene 9.5 M
  Gaussians.
- **Referencia de mesh.** Hacer que coincidieran los renders raster y de path
  trace del mesh bajó el relMSE del raster frente al path tracer de 7.28 a
  3.67 (960×540).

Medido nube frente a mesh (`--no-bake`, cámara del ala, alfa sin dome,
relMSE con él), después de que la coverage pasara a seguir la opacidad:

| | alfa antes → después (mesh) | relMSE antes → después |
|---|---|---|
| ala del gorrión, raster | 0.2395 → 0.2367 (0.2253) | 0.1742 → 0.1635 |
| ala del gorrión, trazada | 0.2385 → 0.2358 (0.2462) | 0.1331 → 0.1247 |

![Figura 15. El gorrión: el mesh con path trace (izquierda) y las Gaussians convertidas, rasterizadas (derecha).](figures/img-sparrow.jpg)

Para la web, el pájaro se convirtió con `--skinned --transfer
--resolution 512`, con bake de lobes zonales a 256 paths y 3 bounces en la
bind pose y la visibilidad de 12 partes añadida (`sparrow_tx_vis.usdc`:
1 504 160 Gaussians, 1 504 096 con contenido). Frente al path trace del mesh
a 1024 paths (1024², `air_fly_A0`, frame 16), el render raster de la nube da
relMSE 0.0025. El cuerpo de la nube sale algo más oscuro y marrón, porque el
transfer zonal solo guarda la luz directa.

## 5.11 El peón: el control rígido

El peón de ajedrez (The Open Chess Set, ASWF, CC BY 4.0) **no tiene
skeleton**. Son dos meshes: un cuerpo de mármol negro `M_Pawn_Body_B` y una
cabeza `M_Pawn_Top_B` con `transmission = 1`, una bola de cristal sobre un
anillo de oro. Está en esta sección porque es el asset que todo cambio
relacionado con la animación tenía que dejar igual:

- Cuando `MeshStageOptions` ganó un tiempo, el peón se convirtió en un fichero
  idéntico byte a byte al de antes.
- Cuando el eje plano pasó a ser
  $\min(\text{along}_U,\text{along}_V)\times0.1$ para corregir el fantasma del
  zorro bajo el ray tracer, el mármol del peón pasó de
  0.0644/0.0798/0.0778 a 0.0645/0.0799/0.0780.
- Toda ruta con skinning, con transformaciones identidad, se reduce a la
  rígida, y el peón es la referencia de la rígida.

Su primera conversión de principio a fin produjo 729 073 Gaussians a
resolución 512. Sus resultados de material y transfer pertenecen a las
secciones 3 y 4. La validación de alta calidad del peón blanco
(`pawn_mesh.usda`, 2048², 2048 paths de referencia) da, por material:

| material | Gaussians | relMSE | p99 | media (nube) | media (path trace del mesh) |
|---|---|---|---|---|---|
| `M_Pawn_Body_W` | 1 568 499 | 0.0218 | 0.273 | 0.167 / 0.165 / 0.133 | 0.164 / 0.162 / 0.132 |
| `M_Pawn_Top_W` (cristal) | 307 296 | 0.1418 | 0.354 | 0.236 / 0.260 / 0.221 | 0.365 / 0.397 / 0.279 |

Tres conversiones posteriores del mismo peón, medidas a 1024², dan un relMSE
del cristal de 0.149, 0.081 y 0.043, con el cuerpo sin cambios en 0.025. El
cuerpo queda a unos pocos por ciento del path trace del mesh. El error se
concentra en la cabeza de cristal (figura 16).

![Figura 16. El peón blanco a 2048²: el path trace del mesh con 2048 paths por píxel (izquierda) y las Gaussians convertidas (derecha).](figures/img-pawn.jpg)

## 5.12 Aplicación: reproducción en la web y en un teléfono

El player web (`sparkwebGPU`, un fork WebGPU de Spark) reutiliza los kernels
de athenea compilados a WGSL. El escritor de `.athc` de athenea rechaza
`--skinned`, así que el lado web lee la nube USD (`usd-athc`, sobre el crate
`openusd`) y escribe un `.athc` v3. Ese fichero añade una sección `SKIN` por
elemento, con las influencias empaquetadas como joint u16 | peso unorm16 y los
gradientes como halves, y un blob de skeleton `ATSK` con los nombres de los
joints, la transformación de bind y el primer clip. Cada uno de los demás
clips es un fichero `ATCL` que se descarga al reproducirlo. Los nodos LoD
resultado de un merge llevan la mezcla de las influencias de sus splats
ponderada por opacidad (las cuatro más pesadas, renormalizadas) y ningún
gradiente.

El kernel de skinning (`athenea_adapter/skin.slang`) reescribe `splatSkin`
línea a línea, porque el original enlaza once storage buffers y Chrome en un
Mac permite diez por etapa. Una prueba de GPU ejecuta el kernel propio de
athenea sobre los mismos splats y compara los dos. Frente al raster de
athenea, el resultado es un relMSE de 0.0015–0.0053 con una razón de brillo
de 1.01. No se escribe movimiento de shutter.

| página | contenido | medido |
|---|---|---|
| `/pawn` (publicada 2026-10-06) | cuerpo 727 061 + cabeza 673 711 splats, transfer, 6 HDRI CC0 | 60 fps en Chrome; conjunto de datos v8: descarga 281.3 → 82.7 MB; memoria de GPU en escritorio 1127 → 800 MB |
| `/sparrow` (publicada 2026-10-07) | 1 504 096 splats, 125.7 MB, 12 clips (6.4 MB gzip), transfer zonal, HDRI | memoria de GPU en escritorio 1084 → 684 MB |
| `/sparrow` en teléfonos | `sparrow-mobile`: 488 656 splats, 31.7 MB, cortado del tree LoD (`--keep-splats 600000`) | 60 fps headless en modo móvil, animado |
| sparrow-v2 (candidato) | 1 100 334 splats (−26.8 %), 75.9 MB; móvil 371 138, 26.1 MB | mismo error que el publicado (4.46e-3) |

**La ruta compacta.** Una nube con skinning también se reproduce por la ruta
compacta de §7: cada nodo lleva su skin en el registro compacto, el dibujo le
aplica el skinning, la caja de una page es la unión de sus cajas en todas las
poses, y los culls de oclusión y de caras traseras se apagan en lo que se mueve.
El merge tree del gorrión se parte en dos por raíces con escalas zonales
comunes, dos ficheros de unos 162 MiB con los clips y sus propios domos, 347 MB
en total (`sparrow-compact-tx`, publicado el 2026-10-18); en el escritorio dibuja
795 k Gaussians a $\tau$ 1.5 y 60 fps en vuelo, y el móvil emulado 232 k.

**Antialiasing temporal.** Se probó en el player una resolución temporal (el
esquema de Karis: jitter de Halton, reprojection, clamp en YCoCg y una media
exponencial), con vectores de movimiento por Gaussian, los de las alas a partir
de las matrices de hueso del frame anterior. Los vectores son correctos (error de
reprojection 0.05 → 0.015–0.028 en las alas, 0.018 → 0.003 en el coche), pero en
movimiento el frame no mejora al frame sin ella en el coche, el gorrión ni la
ruta compacta: muestra más error, parpadeo y estelas, porque los brillos son HDR
y dependen de la vista y el jitter no se promedia. Las variantes (historia
Catmull–Rom, el recorte por varianza de Salvi, un $\alpha$ según la velocidad,
afilado adaptativo al contraste) no lo cambian. Con la cámara quieta iguala una
acumulación de 16 frames. Cuesta 3 ms a 1080p y queda como opción.

**Podar un asset que se mueve.** Las Gaussians ocultas en la bind pose no se
pueden quitar sin más, porque otras poses las muestran. Medido sobre 12
clips × 4 muestras = 48 poses más la bind pose, con 61 vistas cada una: el
61.2 % del gorrión está oculto en la bind pose, pero solo el **32.5 %** lo
está en todas las poses. Quitar lo que oculta la bind pose cuesta un relMSE
de hasta $4.5\times10^{-2}$ en vistas cercanas de poses no muestreadas. Quitar
lo que ocultan todas las poses cuesta como mucho $2.9\times10^{-4}$.
sparrow-v2 es lo segundo. Como comparación, el peón estático solo tiene un
2.7 % oculto.

## 5.13 Lo que no está hecho

- Más de cuatro influencias por Gaussian. Esto limita la precisión en las
  puntas de las alas del gorrión, tanto en la animación como en la
  coincidencia del bake con el mesh.
- El skinning por cuaterniones duales para nubes. El retargeting y la mezcla
  de clips, que una nube con transformaciones de bake no puede hacer.
- Varios meshes con skinning con transformaciones de prim distintas (solo se
  conserva la del primero), y skeletons con transformación animada propia.
- La ruta trazada posa una vez por frame, así que no aplica motion blur a una
  nube con skinning. Las instancias LoD no llevan movimiento.
- El término de autosombra propio de una parte. La oclusión entre partes en
  poses distintas de la bind pose, dentro del transfer zonal. Los bakes de
  visibilidad por encima de unos 3 M Gaussians por parte.
- El error del ajuste zonal en el gorrión, y una medida geométrica de la
  coverage en una flexión vista de cerca, que es lo que mostraría la ganancia
  del término elástico del Jacobian.
